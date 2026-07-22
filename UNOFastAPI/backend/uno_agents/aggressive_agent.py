"""

"""

from backend.uno_agents.base_agent import BaseUNOAgent


class AggressiveAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Aggressive Agent")

    def step(self, state):
        print("")