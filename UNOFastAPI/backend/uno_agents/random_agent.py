"""
This agent selects a random legal move every turn
"""

from backend.uno_agents.base_agent import BaseUNOAgent
import numpy as np


class RandomAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Random Agent")

    def step(self, state):
        # Obtain all legal actions
        legal_actions = self.legal_actions(state)

        # If there are no legal actions, then draw is the only legal action
        if 'draw' in legal_actions:
            return 'draw'

        # Chooses a random card from the list of cards
        action = np.random.choice(legal_actions)
        return action