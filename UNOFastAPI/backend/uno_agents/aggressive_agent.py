"""
This agent prioritizes using special cards first before numbered cards and prefers selecting the least common colour

PRIORITY: wild_draw_4 -> draw_2 -> reverse -> skip -> wild -> numbered
"""

from backend.uno_agents.base_agent import BaseUNOAgent


class AggressiveAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Aggressive Agent")
        self.card_priority = {
            "wild_draw_4" : 6,
            "draw_2" : 5,
            "reverse" : 4,
            "skip" : 3,
            "wild" : 2,
            "number" : 1
        }

    def step(self, state):
        # Obtain legal moves
        legal_actions = self.legal_actions(state)
        hand = self.hand(state)

        if "draw" in legal_actions:
            return "draw"

        least_colour = self.least_common_colour(hand)

        best_action = None
        best_priority = -1

        # Loop over legal actions, choose the card with the highest priority and the least colour
        for action in legal_actions:
            priority = self.card_priority.get(self.action_type(action), 0)

            if priority > best_priority:
                best_priority = priority
                best_action = action

            elif priority == best_priority and self.colour_of_card(action) == least_colour:
                best_action = action

        # When action is a wild card, prefer selecting the most common colour
        if best_action is not None:
            best_colour = self.most_common_colour(hand)

            if best_action.endswith("wild"):
                best_action = f"{best_colour}-wild"

            if best_action.endswith("wild_draw_4"):
                best_action = f"{best_colour}-wild_draw_4"

        return best_action
