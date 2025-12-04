import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import random

# TO-DO
# 1. Implement POMDP Solver
# 2. Belief tracking and card counting
# 3. Implementing further complexity (i.e. doubling / splitting)

"""
What we have currently...

1. BlackjackPOMDP Environment:
   - Full game mechanics with proper ace handling (ace = 11 unless bust, then = 1)
   - Finite deck tracking (cards removed as dealt)
   - Returns observations (visible info) separate from full state
   - Single deck by default, configurable to multi-deck

2. BasicStrategy Policy:
   - Baseline policy using standard blackjack basic strategy
   - Assumes infinite deck (doesn't track deck composition)
   - Handles both soft hands (usable ace) and hard hands
   - Performance: ~8.75% house edge with current rules

3. Value Iteration Solver:
   - Computes optimal policy via dynamic programming
   - Bellman equation: V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]
   - Assumes infinite deck (each card drawn independently)
   - Converges in ~7 iterations
   - Generates both V(s) and Q(s,a) for analysis

4. Monte Carlo Evaluation:
   - Empirical policy evaluation via simulation
   - Runs N episodes and computes: E[R] = (1/N) * Σ rewards
   - Provides metrics: expected value, win/loss/push rates
   - Used to compare policy performance on finite deck games

5. BeliefState Class:
   - Framework for tracking probability distribution over deck composition
   - Bayesian update: P(card_i | deck) = count_i / total_cards
   - Currently defined but NOT YET USED in any policy
   - Will be necessary for POMDP solver implementation
"""


class BlackjackPOMDP:
    # POMDP formulation for blackjack
    # state includes deck composition (hidden), player/dealer hands
    # observations are just what you can see at the table

    '''
    Key idea behind project:
    - In BlackJack, the exact composition of the remaining deck is hidden. Players have to observe
    cards as they're dealth, but must maintain a belief about what remains.
    '''

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        
        # standard deck: 4 of each rank, except 10s (10,J,Q,K all count as 10)
        self.cards_per_deck = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 
                               7: 4, 8: 4, 9: 4, 10: 16}
        self.initial_deck = {}
        for card, count in self.cards_per_deck.items():
            self.initial_deck[card] = count * num_decks
        
        self.HIT = 0
        self.STAND = 1
        
        self.reset()
    
    def reset(self):
        # start a new game
        self.deck = self.initial_deck.copy()
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        
        # initial deal: player gets 2, dealer gets 2
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        
        return self._get_observation()
    
    def _draw_card(self):
        # draw random card from remaining deck
        available_cards = [card for card, count in self.deck.items() if count > 0]
        if not available_cards:
            raise ValueError("Deck is empty")
        
        card = random.choice(available_cards)
        self.deck[card] -= 1
        return card
    
    def _hand_value(self, hand: List[int]):
        # aces count as 11 unless it would bust, then they count as 1
        value = sum(hand)
        num_aces = hand.count(1)
        usable_ace = False
        
        # try to use aces as 11 (adds 10 since already counted as 1)
        for _ in range(num_aces):
            if value + 10 <= 21:
                value += 10
                usable_ace = True
                break  # only use one ace as 11
        
        return value, usable_ace
    
    def _get_observation(self):
        # what the player can actually see
        player_value, usable_ace = self._hand_value(self.player_hand)
        dealer_showing = self.dealer_hand[0]
        
        return {
            'player_value': player_value,
            'dealer_showing': dealer_showing,
            'usable_ace': usable_ace,
            'player_hand': self.player_hand.copy(),
            'deck_state': self.deck.copy()
        }
    
    def step(self, action: int):
        if self.done:
            raise ValueError("Episode already finished")
        
        reward = 0
        
        if action == self.HIT:
            self.player_hand.append(self._draw_card())
            player_value, _ = self._hand_value(self.player_hand)
            
            # check for bust
            if player_value > 21:
                reward = -1
                self.done = True
            
        elif action == self.STAND:
            # dealer's turn now
            reward = self._dealer_play()
            self.done = True
        
        return self._get_observation(), reward, self.done
    
    def _dealer_play(self):
        # dealer must hit on 16 or less, stand on 17+
        while True:
            dealer_value, _ = self._hand_value(self.dealer_hand)
            
            if dealer_value > 21:
                return 1  # dealer busts, we win
            elif dealer_value >= 17:
                break
            else:
                self.dealer_hand.append(self._draw_card())
        
        # compare final hands
        player_value, _ = self._hand_value(self.player_hand)
        dealer_value, _ = self._hand_value(self.dealer_hand)
        
        if player_value > dealer_value:
            return 1
        elif player_value < dealer_value:
            return -1
        else:
            return 0  # tie


class BeliefState:
    # tracks probability distribution over deck composition
    # this is the key to POMDP - maintaining belief about hidden state
    
    def __init__(self, initial_deck: Dict[int, int]):
        self.deck_belief = initial_deck.copy()
        self.total_cards = sum(initial_deck.values())
    
    def update_belief(self, observed_card: int):
        # bayesian update when we see a card
        # P(card_i | deck) = count_i / total_cards
        if self.deck_belief[observed_card] > 0:
            self.deck_belief[observed_card] -= 1
            self.total_cards -= 1
    
    def get_card_probability(self, card: int) -> float:
        if self.total_cards == 0:
            return 0.0
        return self.deck_belief[card] / self.total_cards
    
    def sample_card(self) -> int:
        available_cards = [card for card, count in self.deck_belief.items() if count > 0]
        weights = [self.deck_belief[card] for card in available_cards]
        return random.choices(available_cards, weights=weights)[0]


class BasicStrategy:
    # baseline policy - doesn't track deck composition
    # just uses fixed rules based on infinite deck assumption
    
    def get_action(self, player_value: int, dealer_showing: int, usable_ace: bool) -> int:
        # standard basic strategy with soft/hard hands
        if usable_ace:
            # soft hand (ace counts as 11)
            if player_value >= 19:
                return 1  # STAND on soft 19+
            elif player_value == 18:
                # soft 18: stand vs 2-8, hit vs 9-10-A
                if dealer_showing >= 9 or dealer_showing == 1:
                    return 0  # HIT
                return 1  # STAND
            else:
                return 0  # HIT on soft 17 or less
        else:
            # hard hand
            if player_value >= 17:
                return 1  # STAND
            elif player_value <= 11:
                return 0  # HIT
            else:  # 12-16 is the tricky range
                if dealer_showing >= 7:
                    return 0  # HIT
                elif dealer_showing <= 3:
                    return 0  # HIT
                else:
                    return 1  # STAND


def monte_carlo_evaluation(policy, num_episodes=100000, num_decks=1):
    # evaluate policy by running many simulations
    total_reward = 0
    wins = 0
    losses = 0
    pushes = 0
    
    for _ in range(num_episodes):
        env = BlackjackPOMDP(num_decks=num_decks)
        obs = env.reset()
        episode_reward = 0
        
        while not env.done:
            action = policy.get_action(
                obs['player_value'],
                obs['dealer_showing'],
                obs['usable_ace']
            )
            obs, reward, done = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            pushes += 1
    
    expected_value = total_reward / num_episodes
    win_rate = wins / num_episodes
    
    return {
        'expected_value': expected_value,
        'win_rate': win_rate,
        'loss_rate': losses / num_episodes,
        'push_rate': pushes / num_episodes
    }


class ValueIteration:
    # solve for optimal policy using dynamic programming
    # Bellman equation: V*(s) = max_a sum over s' of P(s'|s,a)[R + gamma*V*(s')]
    
    def __init__(self, gamma=1.0, theta=0.0001):
        self.gamma = gamma  # discount factor
        self.theta = theta  # convergence threshold
        self.V = {}
        self.Q = {}
        
    def evaluate_policy(self, policy, max_iterations=1000, verbose=True):
        # compute value function for a given policy
        self.V = defaultdict(float)
        states = self._generate_states()
        
        for iteration in range(max_iterations):
            delta = 0
            
            for state in states:
                player_sum, dealer_showing, usable_ace = state
                
                if player_sum > 21:
                    continue  # terminal state
                
                v = self.V[state]
                action = policy.get_action(player_sum, dealer_showing, usable_ace)
                new_v = self._compute_action_value(state, action)
                
                self.V[state] = new_v
                delta = max(delta, abs(v - new_v))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: delta = {delta:.6f}")
            
            # check convergence
            if delta < self.theta:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return self.V
    
    def solve_optimal(self, max_iterations=1000, verbose=True):
        # find optimal policy via value iteration
        self.V = defaultdict(float)
        self.Q = defaultdict(lambda: defaultdict(float))
        
        states = self._generate_states()
        
        for iteration in range(max_iterations):
            delta = 0
            
            for state in states:
                player_sum, dealer_showing, usable_ace = state
                
                if player_sum > 21:
                    continue
                
                v = self.V[state]
                
                # compute Q for both actions
                q_hit = self._compute_action_value(state, 0)
                q_stand = self._compute_action_value(state, 1)
                
                self.Q[state][0] = q_hit
                self.Q[state][1] = q_stand
                
                # take max
                self.V[state] = max(q_hit, q_stand)
                
                delta = max(delta, abs(v - self.V[state]))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: delta = {delta:.6f}")
            
            if delta < self.theta:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        return self.V, self.Q
    
    def _generate_states(self):
        states = []
        # state space: player sum (4-21), dealer showing (1-10), usable ace
        for player_sum in range(4, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in [True, False]:
                    # skip invalid states
                    if usable_ace and player_sum < 12:
                        continue  # can't have usable ace with sum < 12
                    if usable_ace and player_sum > 21:
                        continue  # if ace is usable, shouldn't be over 21
                    states.append((player_sum, dealer_showing, usable_ace))
        return states
    
    def _compute_action_value(self, state, action):
        # Q(s,a) = expected value of taking action a in state s
        player_sum, dealer_showing, usable_ace = state
        
        if action == 1:  # STAND
            return self._expected_reward_stand(player_sum, dealer_showing)
        
        else:  # HIT
            expected_value = 0
            
            # infinite deck assumption - each card equally likely
            card_probs = {
                1: 1/13, 2: 1/13, 3: 1/13, 4: 1/13, 5: 1/13,
                6: 1/13, 7: 1/13, 8: 1/13, 9: 1/13, 10: 4/13
            }
            
            for card, prob in card_probs.items():
                new_sum = player_sum + card
                new_usable_ace = usable_ace
                
                # handle ace logic
                if card == 1:
                    # drawing an ace
                    if player_sum + 11 <= 21:
                        new_sum = player_sum + 11
                        new_usable_ace = True
                    else:
                        new_sum = player_sum + 1
                        new_usable_ace = usable_ace
                else:
                    # drawing non-ace
                    new_sum = player_sum + card
                    if new_sum > 21 and usable_ace:
                        # convert ace from 11 to 1
                        new_sum -= 10
                        new_usable_ace = False
                
                if new_sum > 21:
                    expected_value += prob * (-1)  # bust
                else:
                    next_state = (new_sum, dealer_showing, new_usable_ace)
                    expected_value += prob * (0 + self.gamma * self.V[next_state])
            
            return expected_value
    
    def _expected_reward_stand(self, player_sum, dealer_showing):
        # compute expected reward when standing
        # need to account for all possible dealer outcomes
        dealer_bust_prob = self._dealer_bust_probability(dealer_showing)
        
        expected_reward = 0
        expected_reward += dealer_bust_prob * 1  # we win if dealer busts
        
        # if dealer doesn't bust, compare hands
        for dealer_final in range(17, 22):
            prob = self._dealer_final_hand_prob(dealer_showing, dealer_final)
            
            if dealer_final > 21:
                continue
            elif player_sum > dealer_final:
                expected_reward += prob * 1
            elif player_sum < dealer_final:
                expected_reward += prob * (-1)
            # ties contribute 0
        
        return expected_reward
    
    def _dealer_bust_probability(self, showing):
        # approximate probabilities from blackjack theory
        bust_probs = {
            1: 0.12, 2: 0.35, 3: 0.37, 4: 0.40, 5: 0.42,
            6: 0.42, 7: 0.26, 8: 0.24, 9: 0.23, 10: 0.21
        }
        return bust_probs.get(showing, 0.25)
    
    def _dealer_final_hand_prob(self, showing, final_value):
        # simplified - in reality would compute this exactly
        if final_value > 21:
            return 0
        
        remaining_prob = 1 - self._dealer_bust_probability(showing)
        
        # rough approximation: distribute evenly among 17-21
        if 17 <= final_value <= 21:
            return remaining_prob / 5
        return 0
    
    def get_optimal_policy(self):
        if not self.Q:
            raise ValueError("Must run solve_optimal first")
        
        policy = OptimalPolicy(self.Q)
        return policy


class OptimalPolicy:
    def __init__(self, Q):
        self.Q = Q
    
    def get_action(self, player_value, dealer_showing, usable_ace):
        state = (player_value, dealer_showing, usable_ace)
        
        if state not in self.Q:
            # fallback for unseen states
            if player_value >= 17:
                return 1
            return 0
        
        # pick action with highest Q-value
        q_values = self.Q[state]
        return max(q_values.keys(), key=lambda a: q_values[a])


if __name__ == "__main__":
    print("Blackjack POMDP Environment")
    print("=" * 50)
    
    # evaluate basic strategy with monte carlo
    print("\n1. MONTE CARLO EVALUATION")
    print("-" * 50)
    basic_strategy = BasicStrategy()
    results = monte_carlo_evaluation(basic_strategy, num_episodes=10000, num_decks=1)
    
    print("Basic Strategy Performance (100,000 episodes):")
    print(f"Expected Value: {results['expected_value']:.4f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Loss Rate: {results['loss_rate']:.2%}")
    print(f"Push Rate: {results['push_rate']:.2%}")
    
    # now try value iteration
    print("\n2. VALUE ITERATION")
    print("-" * 50)
    vi = ValueIteration(gamma=1.0, theta=0.0001)
    
    print("\nEvaluating Basic Strategy with Value Iteration:")
    V_basic = vi.evaluate_policy(basic_strategy, max_iterations=100, verbose=True)
    
    # look at some interesting states
    print("\nSample State Values (Basic Strategy):")
    sample_states = [
        (20, 10, False),  # hard 20 vs 10
        (16, 10, False),  # hard 16 vs 10 - always tough
        (12, 6, False),   # hard 12 vs 6
        (18, 9, True),    # soft 18 vs 9 (ace counts as 11)
    ]
    for state in sample_states:
        print(f"  State {state}: V = {V_basic[state]:.4f}")
    
    print("\nFinding Optimal Policy with Value Iteration:")
    V_optimal, Q_optimal = vi.solve_optimal(max_iterations=100, verbose=True)
    optimal_policy = vi.get_optimal_policy()
    
    print("\nSample State Values (Optimal Policy):")
    for state in sample_states:
        q_hit = Q_optimal[state][0]
        q_stand = Q_optimal[state][1]
        optimal_action = "HIT" if q_hit > q_stand else "STAND"
        print(f"  State {state}: V = {V_optimal[state]:.4f} | Best action: {optimal_action}")
        print(f"    Q(HIT) = {q_hit:.4f}, Q(STAND) = {q_stand:.4f}")
    
    # compare the two policies
    print("\n3. COMPARING POLICIES")
    print("-" * 50)
    optimal_results = monte_carlo_evaluation(optimal_policy, num_episodes=100000, num_decks=1)
    
    print("Optimal Policy Performance (10,000 episodes):")
    print(f"Expected Value: {optimal_results['expected_value']:.4f}")
    print(f"Win Rate: {optimal_results['win_rate']:.2%}")
    
    improvement = optimal_results['expected_value'] - results['expected_value']
    print(f"\nImprovement over Basic Strategy: {improvement:.4f}")
    
    # play one game to see it in action
    print("\n" + "=" * 50)
    print("Example Game with Optimal Policy:")
    env = BlackjackPOMDP(num_decks=1)
    obs = env.reset()
    
    print(f"Player hand: {obs['player_hand']} (value: {obs['player_value']})")
    print(f"Dealer showing: {obs['dealer_showing']}")
    
    while not env.done:
        action = optimal_policy.get_action(
            obs['player_value'],
            obs['dealer_showing'],
            obs['usable_ace']
        )
        action_name = "HIT" if action == 0 else "STAND"
        print(f"\nAction: {action_name}")
        
        obs, reward, done = env.step(action)
        
        if action == 0 and not done:
            print(f"Player hand: {obs['player_hand']} (value: {obs['player_value']})")
    
    print(f"\nFinal result: {reward} ({'WIN' if reward > 0 else 'LOSS' if reward < 0 else 'PUSH'})")
    print(f"Player final: {obs['player_value']}")
    dealer_value, _ = env._hand_value(env.dealer_hand)
    print(f"Dealer final: {dealer_value}")