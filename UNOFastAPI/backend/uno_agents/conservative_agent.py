"""

"""

from backend.uno_agents.base_agent import BaseUNOAgent


class ConservativeAgent(BaseUNOAgent):

    def __init__(self):
        super().__init__("Conservative Agent")

    def step(self, state):
        print("")