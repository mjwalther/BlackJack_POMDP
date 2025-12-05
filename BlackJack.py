"""
Blackjack RL Project
CS238 Decision Making Under Uncertainty

Authors: Matthias Walther, Diego Padilla, George Song
Dec 2024

We implement Value Iteration, Q-Learning, and SARSA to learn blackjack.
The main extension is tracking deck composition to enable card-counting behavior.

Algorithms based on:
- Value Iteration: Alg 7.8 from textbook
- Q-Learning: Alg 17.2
- SARSA: Alg 17.3
- Epsilon-greedy: Alg 15.3
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random
from abc import ABC, abstractmethod
import pickle

# Make matplotlib optional (for systems with installation issues)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")
    print("To install: pip install matplotlib")


# =============================================================================
# Constants
# =============================================================================

# Actions
HIT = 0
STAND = 1
ACTION_NAMES = {HIT: "HIT", STAND: "STAND"}

# Hyperparameters
GAMMA = 0.95                    # Discount factor
ALPHA = 0.1                     # Learning rate for Q-Learning/SARSA
EPSILON_START = 0.3             # Initial exploration rate
EPSILON_END = 0.01              # Final exploration rate
EPSILON_DECAY = 0.9995          # Exponential decay per episode
NUM_TRAINING_EPISODES = 50000   # Training episodes for model-free methods
NUM_TEST_EPISODES = 10000       # Evaluation episodes
VI_MAX_ITERATIONS = 1000        # Max iterations for Value Iteration
VI_CONVERGENCE_THRESHOLD = 1e-4 # Bellman residual threshold

# Deck bin thresholds (for single deck - 52 cards)
# Low cards (2-6): 20 total in single deck
LOW_CARDS_THRESHOLDS = [10, 15]     # <=10 (ten-rich), 11-15 (neutral), >=16 (ten-poor)
# High cards (10,J,Q,K,A): 20 total in single deck (16 tens + 4 aces)
HIGH_CARDS_THRESHOLDS = [10, 16]    # <=10 (depleted), 11-16 (neutral), >=17 (rich)
# Aces: 4 total in single deck
ACE_THRESHOLDS = [1, 3]             # <=1 (few), 2-3 (normal), >=4 (all present)


# =============================================================================
# BLACKJACK ENVIRONMENT WITH DECK-AWARE STATE FEATURES
# =============================================================================

class BlackjackEnvironment:
    """
    Blackjack game environment with deck tracking.
    
    State is a tuple: (player_sum, dealer_upcard, usable_ace, low_bin, high_bin, ace_bin)
    
    The deck bins discretize the remaining cards to keep state space manageable
    while still capturing card-counting info. Based on Hi-Lo counting system.
    """
    
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        
        # Standard deck: 4 of each rank per deck
        # Cards 2-9 have face value, 10/J/Q/K all count as 10, Ace = 1 or 11
        # We represent cards by their value: 1=Ace, 2-9=face value, 10=ten/face
        self.cards_per_deck = {
            1: 4,   # Aces
            2: 4, 3: 4, 4: 4, 5: 4, 6: 4,  # Low cards (2-6)
            7: 4, 8: 4, 9: 4,              # Neutral cards (7-9)
            10: 16  # Tens, Jacks, Queens, Kings
        }
        
        # Create initial full deck
        self.initial_deck = {}
        for card, count in self.cards_per_deck.items():
            self.initial_deck[card] = count * num_decks
        
        self.reset()
    
    def reset(self) -> Tuple:
        """Start a new game, return initial state."""
        self.deck = self.initial_deck.copy()
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        
        # Initial deal: player gets 2 cards, dealer gets 2 cards
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        
        return self._get_state()
    
    def _draw_card(self) -> int:
        """Draw a random card from the remaining deck."""
        available = [(card, count) for card, count in self.deck.items() if count > 0]
        if not available:
            raise ValueError("Deck is empty!")
        
        total_cards = sum(count for _, count in available)
        cards = [card for card, _ in available]
        weights = [count / total_cards for _, count in available]
        
        card = random.choices(cards, weights=weights)[0]
        self.deck[card] -= 1
        return card
    
    def _hand_value(self, hand: List[int]) -> Tuple[int, bool]:
        """
        Compute hand value with proper ace handling.
        Returns (value, usable_ace) where usable_ace indicates if an ace
        is being counted as 11.
        """
        value = sum(hand)
        num_aces = hand.count(1)
        usable_ace = False
        
        # Try to use one ace as 11 (adds 10 since ace already counted as 1)
        for _ in range(num_aces):
            if value + 10 <= 21:
                value += 10
                usable_ace = True
                break  # Only one ace can be usable
        
        return value, usable_ace
    
    def _discretize_deck(self) -> Tuple[int, int, int]:
        """
        Discretize deck composition into bins (Extension #2).
        
        Inspired by the Hi-Lo card counting system:
        - Low cards (2-6): Favor the dealer
        - High cards (10,J,Q,K,A): Favor the player
        - Aces: Tracked separately for blackjack probability
        
        Returns (low_cards_bin, high_cards_bin, ace_bin)
        Each bin is 0, 1, or 2 representing the discretized count.
        """
        # Count remaining low cards (2-6)
        low_count = sum(self.deck.get(c, 0) for c in [2, 3, 4, 5, 6])
        
        # Count remaining high cards (10s and Aces)
        high_count = self.deck.get(10, 0) + self.deck.get(1, 0)
        
        # Count remaining aces
        ace_count = self.deck.get(1, 0)
        
        # Discretize into bins
        def to_bin(count, thresholds):
            if count <= thresholds[0]:
                return 0  # Low/depleted
            elif count <= thresholds[1]:
                return 1  # Normal/neutral
            else:
                return 2  # High/rich
        
        low_bin = to_bin(low_count, LOW_CARDS_THRESHOLDS)
        high_bin = to_bin(high_count, HIGH_CARDS_THRESHOLDS)
        ace_bin = to_bin(ace_count, ACE_THRESHOLDS)
        
        return (low_bin, high_bin, ace_bin)
    
    def _get_state(self) -> Tuple:
        """
        Get current state representation.
        
        State = (player_total, dealer_upcard, usable_ace, 
                 low_cards_bin, high_cards_bin, ace_bin)
        """
        player_value, usable_ace = self._hand_value(self.player_hand)
        dealer_showing = self.dealer_hand[0]  # Only first card is visible
        low_bin, high_bin, ace_bin = self._discretize_deck()
        
        return (player_value, dealer_showing, int(usable_ace), 
                low_bin, high_bin, ace_bin)
    
    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Take an action in the environment.
        
        Args:
            action: HIT (0) or STAND (1)
            
        Returns:
            (next_state, reward, done)
        """
        if self.done:
            raise ValueError("Episode already finished. Call reset().")
        
        reward = 0.0
        
        if action == HIT:
            # Player draws a card
            self.player_hand.append(self._draw_card())
            player_value, _ = self._hand_value(self.player_hand)
            
            if player_value > 21:
                # Player busts
                reward = -1.0
                self.done = True
        
        elif action == STAND:
            # Dealer's turn
            reward = self._dealer_play()
            self.done = True
        
        return self._get_state(), reward, self.done
    
    def _dealer_play(self) -> float:
        """
        Execute dealer's fixed strategy: hit on 16 or less, stand on 17+.
        Returns the reward for the player.
        """
        while True:
            dealer_value, _ = self._hand_value(self.dealer_hand)
            
            if dealer_value > 21:
                return 1.0  # Dealer busts, player wins
            elif dealer_value >= 17:
                break  # Dealer stands
            else:
                self.dealer_hand.append(self._draw_card())
        
        # Compare final hands
        player_value, _ = self._hand_value(self.player_hand)
        dealer_value, _ = self._hand_value(self.dealer_hand)
        
        if player_value > dealer_value:
            return 1.0   # Player wins
        elif player_value < dealer_value:
            return -1.0  # Player loses
        else:
            return 0.0   # Push (tie)
    
    def get_observation(self) -> Dict:
        """Get detailed observation for debugging/analysis."""
        player_value, usable_ace = self._hand_value(self.player_hand)
        return {
            'player_hand': self.player_hand.copy(),
            'player_value': player_value,
            'usable_ace': usable_ace,
            'dealer_showing': self.dealer_hand[0],
            'dealer_hand': self.dealer_hand.copy(),
            'deck_state': self.deck.copy(),
            'done': self.done
        }


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Implements common functionality: Q-table, action selection, policy extraction.
    """
    
    def __init__(self, gamma: float = GAMMA, alpha: float = ALPHA):
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
    
    @abstractmethod
    def update(self, state, action, reward, next_state, next_action=None):
        """Update Q-values based on experience."""
        pass
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.Q[state][action]
    
    def get_value(self, state: Tuple) -> float:
        """Get value of state (max Q over actions)."""
        return max(self.Q[state][HIT], self.Q[state][STAND])
    
    def greedy_action(self, state: Tuple) -> int:
        """Select greedy action (argmax Q)."""
        q_hit = self.Q[state][HIT]
        q_stand = self.Q[state][STAND]
        
        if q_hit > q_stand:
            return HIT
        elif q_stand > q_hit:
            return STAND
        else:
            return random.choice([HIT, STAND])  # Break ties randomly
    
    def epsilon_greedy_action(self, state: Tuple, epsilon: float) -> int:
        """
        ε-greedy action selection (Algorithm 15.3 from textbook).
        With probability ε: random action (exploration)
        With probability 1-ε: greedy action (exploitation)
        """
        if random.random() < epsilon:
            return random.choice([HIT, STAND])
        else:
            return self.greedy_action(state)
    
    def get_action(self, player_value: int, dealer_showing: int, 
                   usable_ace: bool, low_bin: int = 1, high_bin: int = 1, 
                   ace_bin: int = 1) -> int:
        """Get action for given state (interface compatible with policy evaluation)."""
        state = (player_value, dealer_showing, int(usable_ace), 
                 low_bin, high_bin, ace_bin)
        return self.greedy_action(state)
    
    def get_policy(self) -> Dict[Tuple, int]:
        """Extract policy as dictionary mapping states to actions."""
        policy = {}
        for state in self.Q.keys():
            policy[state] = self.greedy_action(state)
        return policy


# =============================================================================
# VALUE ITERATION AGENT (Algorithm 7.8)
# =============================================================================

class ValueIterationAgent(BaseAgent):
    """
    Model-based agent using Value Iteration (Algorithm 7.8 from textbook).
    
    Computes optimal policy via dynamic programming using the Bellman equation:
    V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
    
    This agent builds an explicit transition model for Blackjack.
    """
    
    def __init__(self, gamma: float = GAMMA, 
                 max_iterations: int = VI_MAX_ITERATIONS,
                 theta: float = VI_CONVERGENCE_THRESHOLD):
        super().__init__(gamma=gamma)
        self.max_iterations = max_iterations
        self.theta = theta  # Convergence threshold (Bellman residual)
        self.V = defaultdict(float)  # Value function
    
    def update(self, state, action, reward, next_state, next_action=None):
        """Value Iteration doesn't need online updates."""
        pass
    
    def _generate_states(self) -> List[Tuple]:
        """Generate all valid states in the state space."""
        states = []
        
        # player_total: 4-21 (can't have less than 4 with two cards)
        # dealer_upcard: 1-10
        # usable_ace: 0 or 1
        # low_bin, high_bin, ace_bin: 0, 1, or 2
        
        for player_sum in range(4, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in [0, 1]:
                    for low_bin in range(3):
                        for high_bin in range(3):
                            for ace_bin in range(3):
                                # Skip invalid states
                                if usable_ace == 1 and player_sum < 12:
                                    continue  # Can't have usable ace with sum < 12
                                if usable_ace == 1 and player_sum > 21:
                                    continue
                                
                                states.append((player_sum, dealer_showing, usable_ace,
                                             low_bin, high_bin, ace_bin))
        return states
    
    def _get_card_probabilities(self, low_bin: int, high_bin: int, 
                                ace_bin: int) -> Dict[int, float]:
        """
        Estimate card draw probabilities from discretized deck state.
        
        Since we only have bins, we estimate probabilities using
        representative counts for each bin.
        """
        # Representative counts based on bin (using midpoints)
        low_counts = {0: 7, 1: 12, 2: 17}    # Estimated remaining 2-6 cards
        high_counts = {0: 7, 1: 13, 2: 18}   # Estimated remaining 10s+Aces
        ace_counts = {0: 0.5, 1: 2, 2: 4}    # Estimated remaining Aces
        
        low_total = low_counts[low_bin]      # Cards 2-6
        ace_total = ace_counts[ace_bin]      # Aces
        tens_total = high_counts[high_bin] - ace_total  # 10s only
        tens_total = max(0, tens_total)
        
        # Estimate neutral cards (7-9): 12 in full deck, assume proportional depletion
        # Use average depletion rate
        deck_depletion = 1.0 - (low_total / 20 + (tens_total + ace_total) / 20) / 2
        neutral_total = 12 * (1 - deck_depletion * 0.5)  # 7, 8, 9 (12 cards total)
        neutral_total = max(3, neutral_total)
        
        total_cards = low_total + neutral_total + tens_total + ace_total
        if total_cards <= 0:
            total_cards = 1  # Avoid division by zero
        
        probs = {}
        # Low cards (2-6): each of 5 ranks gets equal share
        for card in [2, 3, 4, 5, 6]:
            probs[card] = (low_total / 5) / total_cards
        
        # Neutral cards (7-9): each of 3 ranks gets equal share
        for card in [7, 8, 9]:
            probs[card] = (neutral_total / 3) / total_cards
        
        # Tens (10, J, Q, K all count as 10)
        probs[10] = tens_total / total_cards
        
        # Aces
        probs[1] = ace_total / total_cards
        
        # Normalize
        total_prob = sum(probs.values())
        if total_prob > 0:
            for card in probs:
                probs[card] /= total_prob
        
        return probs
    
    def _compute_action_value(self, state: Tuple, action: int) -> float:
        """
        Compute Q(s, a) = E[R + γV(s')] for the given state-action pair.
        """
        player_sum, dealer_showing, usable_ace, low_bin, high_bin, ace_bin = state
        
        if action == STAND:
            # Standing leads to terminal state - use expected reward
            return self._expected_reward_stand(player_sum, dealer_showing)
        
        else:  # HIT
            expected_value = 0.0
            card_probs = self._get_card_probabilities(low_bin, high_bin, ace_bin)
            
            for card, prob in card_probs.items():
                if prob <= 0:
                    continue
                
                # Compute new hand value
                new_sum = player_sum
                new_usable_ace = usable_ace
                
                if card == 1:  # Drawing an Ace
                    if player_sum + 11 <= 21:
                        new_sum = player_sum + 11
                        new_usable_ace = 1
                    else:
                        new_sum = player_sum + 1
                else:  # Drawing non-Ace
                    new_sum = player_sum + card
                    if new_sum > 21 and usable_ace == 1:
                        # Convert usable ace from 11 to 1
                        new_sum -= 10
                        new_usable_ace = 0
                
                if new_sum > 21:
                    # Bust - immediate reward of -1
                    expected_value += prob * (-1.0)
                else:
                    # Continue with discounted future value
                    # Deck state might change slightly, but we keep same bins for simplicity
                    next_state = (new_sum, dealer_showing, new_usable_ace,
                                low_bin, high_bin, ace_bin)
                    expected_value += prob * (0 + self.gamma * self.V[next_state])
            
            return expected_value
    
    def _expected_reward_stand(self, player_sum: int, dealer_showing: int) -> float:
        """
        Compute expected reward when standing with given hand.
        Uses dealer outcome probabilities from standard Blackjack analysis.
        """
        # Dealer bust probabilities (from Blackjack theory)
        bust_probs = {
            1: 0.12, 2: 0.35, 3: 0.37, 4: 0.40, 5: 0.42,
            6: 0.42, 7: 0.26, 8: 0.24, 9: 0.23, 10: 0.21
        }
        
        dealer_bust_prob = bust_probs.get(dealer_showing, 0.25)
        expected_reward = dealer_bust_prob * 1.0  # Win if dealer busts
        
        # Dealer final hand distribution (simplified)
        remaining_prob = 1 - dealer_bust_prob
        dealer_finals = {17: 0.2, 18: 0.2, 19: 0.2, 20: 0.2, 21: 0.2}
        
        for dealer_final, prob_given_no_bust in dealer_finals.items():
            prob = remaining_prob * prob_given_no_bust
            
            if player_sum > dealer_final:
                expected_reward += prob * 1.0   # Win
            elif player_sum < dealer_final:
                expected_reward += prob * (-1.0)  # Lose
            # Ties contribute 0
        
        return expected_reward
    
    def train(self, verbose: bool = True) -> Tuple[Dict, Dict]:
        """
        Run Value Iteration to find optimal policy (Algorithm 7.8).
        
        Returns:
            (V, Q): Value function and Q-function dictionaries
        """
        states = self._generate_states()
        
        if verbose:
            print(f"Value Iteration: {len(states)} states")
        
        for iteration in range(self.max_iterations):
            delta = 0.0
            
            for state in states:
                player_sum = state[0]
                
                if player_sum > 21:
                    continue  # Terminal state
                
                v_old = self.V[state]
                
                # Compute Q for both actions
                q_hit = self._compute_action_value(state, HIT)
                q_stand = self._compute_action_value(state, STAND)
                
                self.Q[state][HIT] = q_hit
                self.Q[state][STAND] = q_stand
                
                # Bellman backup: V(s) = max_a Q(s,a)
                self.V[state] = max(q_hit, q_stand)
                
                delta = max(delta, abs(v_old - self.V[state]))
            
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: Bellman residual = {delta:.6f}")
            
            # Check convergence
            if delta < self.theta:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        return dict(self.V), dict(self.Q)


# =============================================================================
# Q-LEARNING AGENT (Algorithm 17.2)
# =============================================================================

class QLearningAgent(BaseAgent):
    """
    Model-free, off-policy TD learning agent (Algorithm 17.2 from textbook).
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Key characteristic: Uses max over next actions (off-policy).
    This allows Q-Learning to learn the optimal policy while following
    an exploratory policy.
    """
    
    def __init__(self, gamma: float = GAMMA, alpha: float = ALPHA):
        super().__init__(gamma=gamma, alpha=alpha)
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_action: int = None) -> float:
        """
        Q-Learning update (Algorithm 17.2 from textbook).
        
        Uses MAX over next actions (off-policy):
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Returns:
            TD error for analysis
        """
        # Current Q-value
        q_current = self.Q[state][action]
        
        # Max Q-value at next state (off-policy: uses best action)
        if reward != 0:  # Terminal state
            max_q_next = 0.0
        else:
            max_q_next = max(self.Q[next_state][HIT], self.Q[next_state][STAND])
        
        # TD target and error
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - q_current
        
        # Update Q-value
        self.Q[state][action] = q_current + self.alpha * td_error
        
        return td_error


# =============================================================================
# SARSA AGENT (Algorithm 17.3)
# =============================================================================

class SARSAAgent(BaseAgent):
    """
    Model-free, on-policy TD learning agent (Algorithm 17.3 from textbook).
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    Key characteristic: Uses actual next action taken (on-policy).
    SARSA learns the value of the policy being followed, including exploration.
    """
    
    def __init__(self, gamma: float = GAMMA, alpha: float = ALPHA):
        super().__init__(gamma=gamma, alpha=alpha)
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_action: int = None) -> float:
        """
        SARSA update (Algorithm 17.3 from textbook).
        
        Uses actual next action (on-policy):
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        
        Returns:
            TD error for analysis
        """
        # Current Q-value
        q_current = self.Q[state][action]
        
        # Q-value of actual next action (on-policy)
        if reward != 0 or next_action is None:  # Terminal state
            q_next = 0.0
        else:
            q_next = self.Q[next_state][next_action]
        
        # TD target and error
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_current
        
        # Update Q-value
        self.Q[state][action] = q_current + self.alpha * td_error
        
        return td_error


# =============================================================================
# BASIC STRATEGY (BASELINE)
# =============================================================================

class BasicStrategy:
    """
    Standard Blackjack basic strategy as a baseline.
    Assumes infinite deck (doesn't use deck composition information).
    """
    
    def get_action(self, player_value: int, dealer_showing: int, 
                   usable_ace: bool, *args) -> int:
        """
        Returns optimal action according to basic strategy.
        Ignores deck composition (infinite deck assumption).
        """
        if usable_ace:
            # Soft hand (ace counts as 11)
            if player_value >= 19:
                return STAND
            elif player_value == 18:
                # Soft 18: stand vs 2-8, hit vs 9-10-A
                if dealer_showing >= 9 or dealer_showing == 1:
                    return HIT
                return STAND
            else:
                return HIT  # Hit on soft 17 or less
        else:
            # Hard hand
            if player_value >= 17:
                return STAND
            elif player_value <= 11:
                return HIT
            else:  # 12-16
                if dealer_showing >= 7 or dealer_showing == 1:
                    return HIT
                elif dealer_showing <= 3:
                    return HIT
                else:
                    return STAND  # Stand vs 4-6


class RandomPolicy:
    """Random policy baseline - selects HIT or STAND uniformly at random."""
    
    def get_action(self, *args) -> int:
        return random.choice([HIT, STAND])


class ThresholdPolicy:
    """Simple threshold policy - hit if below threshold, else stand."""
    
    def __init__(self, threshold: int = 17):
        self.threshold = threshold
    
    def get_action(self, player_value: int, *args) -> int:
        if player_value < self.threshold:
            return HIT
        return STAND


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Training loop for model-free agents (Q-Learning and SARSA).
    Implements ε-greedy exploration with decay schedule.
    """
    
    def __init__(self, env: BlackjackEnvironment, agent: BaseAgent,
                 num_episodes: int = NUM_TRAINING_EPISODES,
                 epsilon_start: float = EPSILON_START,
                 epsilon_end: float = EPSILON_END,
                 epsilon_decay: float = EPSILON_DECAY):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the agent using the specified number of episodes.
        
        Returns:
            Training history with learning curves and metrics
        """
        epsilon = self.epsilon_start
        history = {
            'episode_rewards': [],
            'avg_rewards': [],      # Moving average every 1000 episodes
            'epsilons': []
        }
        
        reward_buffer = []
        
        for episode in range(self.num_episodes):
            episode_reward = self._train_episode(epsilon)
            reward_buffer.append(episode_reward)
            history['episode_rewards'].append(episode_reward)
            
            # Compute moving average every 1000 episodes
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(reward_buffer[-1000:])
                history['avg_rewards'].append(avg_reward)
                history['epsilons'].append(epsilon)
                
                if verbose:
                    print(f"  Episode {episode + 1}: avg_reward = {avg_reward:.4f}, "
                          f"ε = {epsilon:.4f}")
            
            # Decay epsilon
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
        
        return history
    
    def _train_episode(self, epsilon: float) -> float:
        """Run one training episode and return total reward."""
        state = self.env.reset()
        total_reward = 0.0
        
        # For SARSA, we need the first action
        action = self.agent.epsilon_greedy_action(state, epsilon)
        
        while not self.env.done:
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            
            if not done:
                next_action = self.agent.epsilon_greedy_action(next_state, epsilon)
            else:
                next_action = None
            
            # Update agent (SARSA uses next_action, Q-Learning ignores it)
            self.agent.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
        
        return total_reward


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class Evaluator:
    """
    Evaluation utilities for comparing agents.
    """
    
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
    
    def evaluate_policy(self, policy, num_episodes: int = NUM_TEST_EPISODES,
                       seed: Optional[int] = None) -> Dict:
        """
        Evaluate a policy by running Monte Carlo simulations.
        
        Args:
            policy: Object with get_action(player_value, dealer_showing, usable_ace, ...) method
            num_episodes: Number of episodes to simulate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with metrics: expected_value, win_rate, loss_rate, push_rate
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        env = BlackjackEnvironment(num_decks=self.num_decks)
        
        total_reward = 0
        wins = 0
        losses = 0
        pushes = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            while not env.done:
                player_val, dealer_show, usable, low_b, high_b, ace_b = state
                action = policy.get_action(player_val, dealer_show, bool(usable),
                                          low_b, high_b, ace_b)
                state, reward, done = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                pushes += 1
        
        return {
            'expected_value': total_reward / num_episodes,
            'win_rate': wins / num_episodes,
            'loss_rate': losses / num_episodes,
            'push_rate': pushes / num_episodes,
            'total_episodes': num_episodes
        }
    
    def analyze_deck_awareness(self, agent: BaseAgent, 
                              num_episodes: int = 5000) -> Dict:
        """
        Analyze how the agent's policy differs based on deck composition.
        Compares behavior in "ten-rich" vs "ten-poor" deck states.
        """
        # States to analyze (key decision points)
        test_states = []
        for player_sum in [12, 13, 14, 15, 16]:  # Tricky range
            for dealer_showing in [2, 7, 10]:
                test_states.append((player_sum, dealer_showing, 0))
        
        results = {
            'ten_rich_actions': {},   # low_bin=0, high_bin=2 (favorable)
            'neutral_actions': {},    # low_bin=1, high_bin=1
            'ten_poor_actions': {},   # low_bin=2, high_bin=0 (unfavorable)
            'policy_differences': []
        }
        
        for base_state in test_states:
            player_sum, dealer_showing, usable_ace = base_state
            
            # Ten-rich deck (favorable for player)
            ten_rich_state = (player_sum, dealer_showing, usable_ace, 0, 2, 1)
            # Neutral deck
            neutral_state = (player_sum, dealer_showing, usable_ace, 1, 1, 1)
            # Ten-poor deck (unfavorable for player)
            ten_poor_state = (player_sum, dealer_showing, usable_ace, 2, 0, 1)
            
            a_rich = agent.greedy_action(ten_rich_state)
            a_neutral = agent.greedy_action(neutral_state)
            a_poor = agent.greedy_action(ten_poor_state)
            
            results['ten_rich_actions'][base_state] = ACTION_NAMES[a_rich]
            results['neutral_actions'][base_state] = ACTION_NAMES[a_neutral]
            results['ten_poor_actions'][base_state] = ACTION_NAMES[a_poor]
            
            # Record if policy differs based on deck state
            if a_rich != a_neutral or a_rich != a_poor:
                results['policy_differences'].append({
                    'state': base_state,
                    'ten_rich': ACTION_NAMES[a_rich],
                    'neutral': ACTION_NAMES[a_neutral],
                    'ten_poor': ACTION_NAMES[a_poor]
                })
        
        return results
    
    def compare_to_vi(self, agent: BaseAgent, vi_agent: ValueIterationAgent) -> Dict:
        """Compare an agent's policy to Value Iteration (optimal)."""
        states_compared = 0
        states_matching = 0
        
        for state in vi_agent.Q.keys():
            vi_action = vi_agent.greedy_action(state)
            agent_action = agent.greedy_action(state)
            
            states_compared += 1
            if vi_action == agent_action:
                states_matching += 1
        
        return {
            'states_compared': states_compared,
            'states_matching': states_matching,
            'match_rate': states_matching / states_compared if states_compared > 0 else 0
        }


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_learning_curves(histories: Dict[str, Dict], save_path: str = None):
    """Plot learning curves for model-free algorithms."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_learning_curves (matplotlib not available)")
        return
    plt.figure(figsize=(9, 5))
    
    colors = {'Q-Learning': 'blue', 'SARSA': 'orange'}
    
    for name, history in histories.items():
        episodes = np.arange(1000, len(history['avg_rewards']) * 1000 + 1, 1000)
        plt.plot(episodes, history['avg_rewards'], label=name, 
                color=colors.get(name, None), linewidth=1.5)
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Avg Reward (per 1k episodes)')
    plt.title('Model-Free Learning Curves')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # reference line at 0
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_performance_comparison(results: Dict[str, Dict], save_path: str = None):
    """Bar chart comparing different algorithms."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_performance_comparison (matplotlib not available)")
        return
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.suptitle('Algorithm Performance Comparison', fontsize=12, y=1.02)
    
    metric_labels = {
        'expected_value': 'Expected Value', 
        'win_rate': 'Win Rate',
        'loss_rate': 'Loss Rate', 
        'push_rate': 'Push Rate'
    }
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']  # seaborn-ish colors
    
    for i, metric in enumerate(['expected_value', 'win_rate', 'loss_rate', 'push_rate']):
        values = [results[alg][metric] for alg in algorithms]
        
        bars = axes[i].bar(algorithms, values, color=colors)
        axes[i].set_ylabel(metric_labels[metric])
        axes[i].tick_params(axis='x', rotation=40)
        axes[i].set_axisbelow(True)
        axes[i].yaxis.grid(True, linestyle='--', alpha=0.5)
        
        # value labels
        for bar, v in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_policy_heatmap(agent: BaseAgent, usable_ace: bool = False, 
                       deck_state: Tuple[int, int, int] = (1, 1, 1),
                       title: str = None, save_path: str = None):
    """
    Shows the learned policy as a heatmap.
    H = hit, S = stand
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot_policy_heatmap (matplotlib not available)")
        return
    low_bin, high_bin, ace_bin = deck_state
    
    # build the policy grid
    player_sums = list(range(12, 22))  # 12-21
    dealer_cards = list(range(1, 11))  # A thru 10
    
    policy_grid = np.zeros((len(player_sums), len(dealer_cards)))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            state = (player_sum, dealer_card, int(usable_ace), 
                    low_bin, high_bin, ace_bin)
            action = agent.greedy_action(state)
            policy_grid[i, j] = action  # 0=HIT, 1=STAND
    
    # plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # simple red/green colormap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(policy_grid, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # axis labels
    ax.set_xticks(range(len(dealer_cards)))
    ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    ax.set_yticks(range(len(player_sums)))
    ax.set_yticklabels(player_sums)
    
    ax.set_xlabel('Dealer Up Card')
    ax.set_ylabel('Player Total')
    
    # put H or S in each cell
    for i in range(len(player_sums)):
        for j in range(len(dealer_cards)):
            txt = 'S' if policy_grid[i, j] == 1 else 'H'
            ax.text(j, i, txt, ha='center', va='center', 
                   color='black', fontsize=9, fontweight='bold')
    
    # title
    hand_type = "Soft" if usable_ace else "Hard"
    if title:
        ax.set_title(f'{title} ({hand_type} hands)')
    else:
        deck_desc = f"deck bins: low={low_bin}, high={high_bin}, ace={ace_bin}"
        ax.set_title(f'Policy - {hand_type} Hands\n({deck_desc})')
    
    # simple legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor='#66bb6a', label='Stand'),
                      Patch(facecolor='#ef5350', label='Hit')]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_experiments(verbose: bool = True, save_results: bool = True):
    """
    Main experiment runner - trains all agents and evaluates them.
    """
    print("=" * 50)
    print("Blackjack RL - CS238 Final Project")
    print("=" * 50)
    
    # Initialize environment
    env = BlackjackEnvironment(num_decks=2)
    evaluator = Evaluator(num_decks=2)
    
    results = {}
    histories = {}
    
    # -------------------------------------------------------------------------
    # Value Iteration
    # -------------------------------------------------------------------------
    print("\n[1/3] Running Value Iteration...")
    
    vi_agent = ValueIterationAgent(gamma=GAMMA)
    vi_agent.train(verbose=verbose)
    
    print("Evaluating...")
    results['Value Iteration'] = evaluator.evaluate_policy(vi_agent)
    print(f"  E[R] = {results['Value Iteration']['expected_value']:.4f}")
    print(f"  Win rate: {results['Value Iteration']['win_rate']:.2%}")
    print(f"  Loss rate: {results['Value Iteration']['loss_rate']:.2%}")
    
    # -------------------------------------------------------------------------
    # Q-Learning  
    # -------------------------------------------------------------------------
    print("\n[2/3] Training Q-Learning agent...")
    
    q_agent = QLearningAgent(gamma=GAMMA, alpha=ALPHA)
    q_trainer = Trainer(env, q_agent, num_episodes=NUM_TRAINING_EPISODES)
    
    histories['Q-Learning'] = q_trainer.train(verbose=verbose)
    
    print("Evaluating...")
    results['Q-Learning'] = evaluator.evaluate_policy(q_agent)
    print(f"  E[R] = {results['Q-Learning']['expected_value']:.4f}")
    print(f"  Win rate: {results['Q-Learning']['win_rate']:.2%}")
    
    # -------------------------------------------------------------------------
    # SARSA
    # -------------------------------------------------------------------------
    print("\n[3/3] Training SARSA agent...")
    
    sarsa_agent = SARSAAgent(gamma=GAMMA, alpha=ALPHA)
    sarsa_trainer = Trainer(BlackjackEnvironment(num_decks=2), sarsa_agent, 
                           num_episodes=NUM_TRAINING_EPISODES)
    
    histories['SARSA'] = sarsa_trainer.train(verbose=verbose)
    
    print("Evaluating...")
    results['SARSA'] = evaluator.evaluate_policy(sarsa_agent)
    print(f"  E[R] = {results['SARSA']['expected_value']:.4f}")
    print(f"  Win rate: {results['SARSA']['win_rate']:.2%}")
    
    # -------------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Evaluating baselines...")
    
    basic = BasicStrategy()
    results['Basic Strategy'] = evaluator.evaluate_policy(basic)
    print(f"  Basic Strategy: E[R] = {results['Basic Strategy']['expected_value']:.4f}")
    
    random_policy = RandomPolicy()
    results['Random'] = evaluator.evaluate_policy(random_policy)
    print(f"  Random: E[R] = {results['Random']['expected_value']:.4f}")
    
    threshold = ThresholdPolicy(threshold=17)
    results['Threshold-17'] = evaluator.evaluate_policy(threshold)
    print(f"  Threshold(17): E[R] = {results['Threshold-17']['expected_value']:.4f}")
    
    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Comparing policies to Value Iteration:")
    q_match = evaluator.compare_to_vi(q_agent, vi_agent)
    sarsa_match = evaluator.compare_to_vi(sarsa_agent, vi_agent)
    print(f"  Q-Learning agreement: {q_match['match_rate']:.1%}")
    print(f"  SARSA agreement: {sarsa_match['match_rate']:.1%}")
    
    # check if agents learned deck-aware behavior
    print("\nDeck-awareness check:")
    deck_analysis = evaluator.analyze_deck_awareness(q_agent)
    n_diff = len(deck_analysis['policy_differences'])
    print(f"  Found {n_diff} states where policy depends on deck composition")
    if n_diff > 0 and n_diff <= 5:
        for diff in deck_analysis['policy_differences']:
            print(f"    {diff['state']}: rich={diff['ten_rich']}, "
                  f"neutral={diff['neutral']}, poor={diff['ten_poor']}")
    elif n_diff > 5:
        for diff in deck_analysis['policy_differences'][:3]:
            print(f"    {diff['state']}: rich={diff['ten_rich']}, "
                  f"neutral={diff['neutral']}, poor={diff['ten_poor']}")
        print(f"    ... and {n_diff - 3} more")
    
    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"\n{'Method':<18} {'E[R]':>8} {'Win':>8} {'Loss':>8} {'Push':>8}")
    print("-" * 50)
    for alg, res in results.items():
        print(f"{alg:<18} {res['expected_value']:>8.4f} "
              f"{res['win_rate']*100:>7.1f}% {res['loss_rate']*100:>7.1f}% "
              f"{res['push_rate']*100:>7.1f}%")
    
    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Generating plots...")
    
    plot_learning_curves(histories, save_path='learning_curves.png')
    
    # Value Iteration policies
    plot_policy_heatmap(vi_agent, usable_ace=False, deck_state=(1, 1, 1),
                       title='Value Iteration (neutral deck)', 
                       save_path='vi_policy_neutral.png')
    
    plot_policy_heatmap(vi_agent, usable_ace=False, deck_state=(0, 2, 1),
                       title='Value Iteration (favorable deck)', 
                       save_path='vi_policy_favorable.png')
    
    plot_policy_heatmap(vi_agent, usable_ace=False, deck_state=(2, 0, 1),
                       title='Value Iteration (unfavorable deck)', 
                       save_path='vi_policy_unfavorable.png')
    
    # Q-Learning policies
    plot_policy_heatmap(q_agent, usable_ace=False, deck_state=(1, 1, 1),
                       title='Q-Learning (neutral deck)',
                       save_path='ql_policy_neutral.png')
    
    plot_policy_heatmap(q_agent, usable_ace=False, deck_state=(0, 2, 1),
                       title='Q-Learning (favorable deck)',
                       save_path='ql_policy_favorable.png')
    
    plot_policy_heatmap(q_agent, usable_ace=False, deck_state=(2, 0, 1),
                       title='Q-Learning (unfavorable deck)',
                       save_path='ql_policy_unfavorable.png')
    
    # SARSA policies
    plot_policy_heatmap(sarsa_agent, usable_ace=False, deck_state=(1, 1, 1),
                       title='SARSA (neutral deck)',
                       save_path='sarsa_policy_neutral.png')
    
    plot_policy_heatmap(sarsa_agent, usable_ace=False, deck_state=(0, 2, 1),
                       title='SARSA (favorable deck)',
                       save_path='sarsa_policy_favorable.png')
    
    plot_policy_heatmap(sarsa_agent, usable_ace=False, deck_state=(2, 0, 1),
                       title='SARSA (unfavorable deck)',
                       save_path='sarsa_policy_unfavorable.png')
    
    # save data
    if save_results:
        with open('results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'histories': histories,
                'deck_analysis': deck_analysis
            }, f)
        print("Results saved to results.pkl")
    
    print("\nDone!")
    
    return results, histories, vi_agent, q_agent, sarsa_agent


def play_demo_game(agent: BaseAgent, verbose: bool = True):
    """Play a demonstration game with the given agent."""
    env = BlackjackEnvironment(num_decks=1)
    state = env.reset()
    obs = env.get_observation()
    
    if verbose:
        print("\n" + "=" * 40)
        print("DEMO GAME")
        print("=" * 40)
        print(f"Player hand: {obs['player_hand']} (value: {obs['player_value']})")
        print(f"Dealer showing: {obs['dealer_showing']}")
        print(f"State: {state}")
    
    total_reward = 0
    while not env.done:
        action = agent.greedy_action(state)
        
        if verbose:
            print(f"\nAction: {ACTION_NAMES[action]}")
        
        state, reward, done = env.step(action)
        total_reward += reward
        obs = env.get_observation()
        
        if verbose and not done:
            print(f"Player hand: {obs['player_hand']} (value: {obs['player_value']})")
    
    if verbose:
        print(f"\nFinal result: {'WIN' if total_reward > 0 else 'LOSS' if total_reward < 0 else 'PUSH'}")
        print(f"Player final: {obs['player_value']}")
        dealer_val, _ = env._hand_value(env.dealer_hand)
        print(f"Dealer final: {dealer_val} (hand: {env.dealer_hand})")
    
    return total_reward


if __name__ == "__main__":
    # Run full experiments
    results, histories, vi_agent, q_agent, sarsa_agent = run_experiments(
        verbose=True, 
        save_results=True
    )
    
    # Play demo games
    print("\n" + "=" * 60)
    print("DEMO GAMES")
    print("=" * 60)
    
    print("\n--- Using Value Iteration Agent ---")
    for i in range(3):
        play_demo_game(vi_agent)
    
    print("\n--- Using Q-Learning Agent ---")
    for i in range(3):
        play_demo_game(q_agent)
