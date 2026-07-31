"""
Balanced Agent heuristics (highest priority to lowest):

1. Threat prevention
    - if any opponent has 2 or fewer cards remaining,
      immediately prioritize attack cards: wild_draw_4 > draw_2 > skip > reverse

2. Colour consolidation
   - prefer playing cards from colours that appear LEAST often in our hand

3. Special card usage
   - when there is no immediate threat, prefer special cards (draw_2, skip, reverse) over number cards

4. Number card usage
   - if multiple number cards are available, prefer higher-valued cards first

5. Wild card preservation
   - save 'wild' and 'wild draw four' cards
   - their flexibility is helpful in difficult situations

6. Wild colour selection
   - when playing a 'wild' or 'wild draw four', choose the colour that appears most often in our hand

7. Draw Only When Necessary
   - if no playable card exists, draw.
"""

from backend.uno_agents.base_agent import BaseUNOAgent


class BalancedAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Balanced Agent")
        self.normal_priority = {
            "draw_2": 5,
            "skip": 4,
            "reverse": 3,
            "number": 2,
            "wild": 1,
            "wild_draw_4": 0
        }

        self.threat_priority = {
            "wild_draw_4": 6,
            "draw_2": 5,
            "skip": 4,
            "reverse": 3,
            "wild": 2,
            "number": 1
        }

    def step(self, state):
        legal_actions = self.legal_actions(state)

        if "draw" in legal_actions:
            return "draw"

        hand = self.hand(state)

        # Threat prevention

        opponent_counts = self.opponent_card_counts(state)

        # Smallest opponent hand size
        opponent_min = min(opponent_counts)

        if opponent_min <= 2:
            priorities = self.threat_priority
        else:
            priorities = self.normal_priority

        # Evaluate legal actions

        best_action = None
        best_score = float("-inf")

        colour_counts = self.count_colours(self.filter_wild(hand))

        for action in legal_actions:
            action_type = self.action_type(action)

            score = priorities.get(action_type, 0) * 100

            # Eliminate weak colours (less common colours get higher score)
            colour = action[0]

            if colour in colour_counts:
                score += (max(colour_counts.values()) - colour_counts[colour])

            # Prefer larger number cards
            if action_type == "number":
                value = int(action.split("-")[1])
                score += value

            if score > best_score:
                best_score = score
                best_action = action

        # Choose best colour for wild cards
        if best_action is not None:
            best_colour = self.most_common_colour(hand)

            if best_action.endswith("wild"):
                best_action = f"{best_colour}-wild"

            if best_action.endswith("wild_draw_4"):
                best_action = f"{best_colour}-wild_draw_4"

        return best_action