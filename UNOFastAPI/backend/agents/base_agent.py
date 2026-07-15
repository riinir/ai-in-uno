class BaseUNOAgent:

    use_raw = True

    def step(self, state):
        raise NotImplementedError

    def eval_step(self, state):
        return self.step(state), []