import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import random

class BlackjackPOMDP:
    """
    Notes:
    - Game is only 1 player and 1 dealer
    - Actions include hit or stand (can implement more later on)
    - Will consider Aces as only 1 for the sake of this project (subject to change if we decide)
    
    POMDP Part:
    - Belief state: probability distribution over remaining deck
    - Transition: depends on unknown deck composition
    - Reward: +1 win, -1 loss, 0 push (tie)
    """
    
    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        # Card values: A=1, 2-9=face, 10/J/Q/K=10
        # We will designate Ace as only 1, rather than 1 or 11, for the sake of this project
        self.cards_per_deck = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 16}  # 10,J,Q,K
        self.initial_deck = {k: v * num_decks for k, v in self.cards_per_deck.items()}
        
        # Actions
        self.HIT = 0
        self.STAND = 1
        # Will implement more actions for complexity (double down and split)
        
        self.reset()


    def reset(self):
        """
        Function: reset(self)

        Purpose: Begins a new game of BlackJack
        """
        self.deck = self.initial_deck.copy()
        self.player_hand = []
        self.dealer_hand = []
        self.game_finished = False

        # Deal starting cards to player and dealer
        self.player_hand.append(self.deal_card())
        self.dealer_hand.append(self.deal_card())
        self.player_hand.append(self.deal_card())
        self.dealer_hand.append(self.deal_card())
        
        return f"Player hand: {self.player_hand} Dealer hand: {self.dealer_hand}"

    def deal_card(self):
        """
        Function: deal_card(self)
        
        Purpose: Drawing a card from the current game deck
        """
        available_cards = []            # List of cards that have more than 0 count left in the deck
        for card, count in self.deck.items():
            if count > 0:
                available_cards.append(card)

        if len(available_cards) == 0:
            raise ValueError("Deck has no more cards.")
        
        card = random.choice(available_cards)       # Choose card from deck
        self.deck[card] -= 1                        # Reduce card count by 1 from deck
        return card


# TESTING
# env = BlackjackPOMDP(num_decks=1)
# state = env.reset()
# print(state)

# print("Player hand list:", env.player_hand)
# print("Dealer hand list:", env.dealer_hand)
# print("Remaining deck:", env.deck)



