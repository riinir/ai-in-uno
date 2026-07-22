"""
Base class for all custom UNO rule-based agents

Every rule-based agent should inherit from BaseUNOAgent and only implement
the step() method

The interface includes many helper and utility functions for child classes
"""

from abc import ABC, abstractmethod
from collections import Counter


class BaseUNOAgent(ABC):
    """
    Required rlcard interface:
        - use_raw
        - step(state)
        - eval_step(state)
    """

    # rlcard should provide raw states and expect raw action strings (e.g., "r-7", not an integer alone)
    use_raw = True

    def __init__(self, name="Base Agent"):
        """
        Names: "Random Agent", "Aggressive Agent", "Conservative Agent", "Balanced Agent"
        """
        self.name = name

    @abstractmethod
    def step(self, state):
        """
        Decide which action to take

        Parameters:
        state : dict, rlcard raw state

        Returns:
        Raw UNO action : str
            Examples:
                "r-5"
                "g-skip"
                "y-draw_2"
                "draw"
                "b-wild_draw_4"
        """
        pass

    def eval_step(self, state):
        """
        rlcard evaluation interface

        reuse step()
        """
        return self.step(state), []

    # Helper functions

    @staticmethod
    def legal_actions(state):
        # Return all legal raw actions
        return state["raw_legal_actions"]

    @staticmethod
    def observation(state):
        # Return raw observation dictionary; the current state of the game
        return state["raw_obs"]

    @staticmethod
    def hand(state):
        # Return player's hand
        return state["raw_obs"]["hand"]

    @staticmethod
    def target(state):
        # Return current target card; latest card on the discard pile
        return state["raw_obs"]["target"]

    @staticmethod
    def opponent_card_counts(state):
        # Return list of opponent hand sizes
        return state["raw_obs"]["num_cards"]

    @staticmethod
    def current_player(state):
        # Return the player number of the current turn (human == 0, agent == 1)
        return state["raw_obs"]["current_player"]

    # Card Utility

    @staticmethod
    def filter_wild(cards):
        # Return a list of all the non-wild cards. Return original list if all are wild
        filtered = [
            card for card in cards
            if "wild" not in card.get_str()
        ]

        return filtered if filtered else cards

    @staticmethod
    def count_colors(cards):
        """
        Count how many cards of each colour exist

        Returns : dict
            Example:
            {
                'r': 4,
                'g': 2,
                'b': 1,
                'y': 5
            }
        """
        counter = Counter()

        for card in cards:
            if len(card) > 0:
                counter[card] += 1

        return dict(counter)

    @staticmethod
    def most_common_color(cards):
        # Return colour (str) that appears most often
        counts = BaseUNOAgent.count_colors(BaseUNOAgent.filter_wild(cards))

        if not counts:
            return "r"

        return max(counts)

    @staticmethod
    def least_common_color(cards):
        # Return colour that appears least often
        counts = BaseUNOAgent.count_colors(BaseUNOAgent.filter_wild(cards))

        if not counts:
            return "r"

        return min(counts)

    @staticmethod
    def action_type(action):
        """
        Return action type

        Examples:
        r-5              -> "number"
        g-skip           -> "skip"
        y-draw_2         -> "draw_2"
        b-reverse        -> "reverse"
        draw             -> "draw"
        r-wild           -> "wild"
        y-wild_draw_4    -> "wild_draw_4"
        """
        if action == "draw":
            return "draw"

        parts = action.split("-")

        if len(parts) < 2:
            return "unknown"

        value = parts[1]

        if value.isdigit():
            return "number"

        return value

    def __repr__(self):
        # String representation of the agent object
        return self.name