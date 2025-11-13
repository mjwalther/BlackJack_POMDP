import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import random

class BlackjackPOMDP:
    """
    Blackjack as a Partially Observable Markov Decision Process

    Notes:
    - Game is only 1 player and 1 dealer
    - Actions include hit or stand (can implement more later on)
    - Will consider Aces as only 1 for the sake of this project (subject to change if we decide)
    
    POMDP aspects:
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