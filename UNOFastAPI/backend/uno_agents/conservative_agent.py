"""
This agent:
- prioritizes using weaker cards
- prioritizes the most common colour in hand
- saves special cards for when an opponent has a low hand

PRIORITY: numbered -> wild -> skip -> reverse -> draw_2 -> wild_draw_4
"""

from backend.uno_agents.base_agent import BaseUNOAgent


class ConservativeAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Conservative Agent")
        self.normal_priority = {
            "number": 6,
            "wild": 5,
            "skip": 4,
            "reverse": 3,
            "draw_2": 2,
            "wild_draw_4": 1
        }

        self.defensive_priority = {
            "wild_draw_4": 6,
            "draw_2": 5,
            "skip": 4,
            "reverse": 3,
            "wild": 2,
            "number": 1
        }

    def step(self, state):
        legal_actions = self.legal_actions(state)
        print(f"ALL LEGAL ACTIONS: {legal_actions}")
        hand = self.hand(state)

        if "draw" in legal_actions:
            return "draw"

        # All opponents' hand sizes
        num_cards = self.opponent_card_counts(state)
        print(f"NUM CARDS: {num_cards}")

        # Smallest opponent hand (excluding ourselves)
        opponent_min = min(num_cards)
        print(f"OPPONENT MIN: {opponent_min}")

        # Switch strategy if an opponent is close to winning
        if opponent_min <= 3:
            priorities = self.defensive_priority
        else:
            priorities = self.normal_priority

        best_action = None
        best_priority = -1

        most_common_colour = self.most_common_colour(hand)

        for action in legal_actions:
            priority = priorities.get(self.action_type(action), 0)

            if priority > best_priority:
                best_priority = priority
                best_action = action

            elif priority == best_priority and self.colour_of_card(action) == most_common_colour:
                best_action = action

        # When action is a wild card, prefer selecting the most common colour
        if best_action is not None:
            best_colour = self.most_common_colour(hand)

            if best_action.endswith("wild"):
                best_action = f"{best_colour}-wild"

            if best_action.endswith("wild_draw_4"):
                best_action = f"{best_colour}-wild_draw_4"


        return best_action