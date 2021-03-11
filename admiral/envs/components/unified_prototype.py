
import numpy as np

def in_bounds(state, agent, proposed_position):
    """
    Return True if the proposed position is in bounds. The bounds is defined
    as [0, region)^N, where N is the spatial dimension.
    """
    return np.all(proposed_position >= 0) and np.all(proposed_position < state.region)

class PositionAgent:
    def __init__(self, initial_position=None, **kwargs):
        self.initial_position = initial_position
        self.position = None


class State:
    def __init__(self, set_reqs=None, **kwargs):
        """
        set_reqs (list of funtions):
            List of functions that return booleans. Each function returns True if
            the proposed change in state is valid. Otherwise, it returns False.
        """
        self.set_reqs = [] if set_reqs is None else set_reqs
    
    def check_reqs(self, agent, proposed_state_change, **kwargs):
        return all([True and func(self, agent, proposed_state_change) for func in self.set_reqs])
    
class PositionState(State):
    def __init__(self, region=None, agents=None, **kwargs):
        self.region = region
        self.agents = agents
    
    def reset(self, **kwargs):
        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                agent.position = None
        
        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                if agent.initial_position is not None:
                    agent.position = agent.initial_position
                else:
                    agent.position = np.random.uniform(region, (2,))
    
    def set_position(self, agent, _position, **kwargs):
        if super().check_reqs(agent, _position):
            agent.position = _position
    
    def modify_position(self, agent, value, **kwargs):
        self.set_position(agent, agent.position + value, **kwargs)


