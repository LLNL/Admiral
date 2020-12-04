
class RewarderComponent:
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}

    def get_reward(self, agent_id, **kwargs):
        return self.rewards[agent_id]