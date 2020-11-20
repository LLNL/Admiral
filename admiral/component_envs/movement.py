
import numpy as np

from admiral.envs import Agent

class GridMovementAgent(Agent):
    def __init__(self, move=None, **kwargs):
        assert move is not None, "move must be an integer"
        self.move = move
        super().__init__(**kwargs)
    
    @property
    def configured(self):
        return super().configured and self.move is not None

class GridMovementEnv:
    """
    Agents in the GridWorld can move around.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer"
        self.region = region
        # assert type(agents) is dict, "agents must be a dict"
        # for agent in agents.values():
        #     assert isinstance(agent, GridMovementAgent)
        # self.agents = agents
        self.agents = {agent.id: agent for agent in agents.values() if isinstance(agent, GridMovementAgent)}

        from gym.spaces import Box
        for agent in self.agents.values():
            agent.action_space['move'] = Box(-agent.move, agent.move, (2,), np.int)

    def process_move(self, position, move, **kwargs):
        """
        Move the agent according to the move action. Returns the proposed new position.
        """
        new_position = position + move
        if 0 <= new_position[0] < self.region and \
           0 <= new_position[1] < self.region: # Still inside the boundary, good move
            return new_position
        else:
            return position
