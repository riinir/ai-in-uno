"""

"""

from backend.uno_agents.base_agent import BaseUNOAgent


class BalancedAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Balanced Agent")

    def step(self, state):
        print("")