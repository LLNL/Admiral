
import numpy as np

from admiral.envs.components.agent import Agent

class ContinuousSpaceAgent(Agent):
    def __init__(self, radius=None, initial_position=None, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.initial_position = initial_position
        self.position = None

class State:
    def __init__(self, reqs=None, **kwargs):
        """
        reqs (list of funtions):
            List of functions that return booleans. Each function returns True if
            the proposed change in state is valid. Otherwise, it returns False.
        """
        self.reqs = [] if reqs is None else reqs
    
    def check_reqs(self, agent, proposed_state_change, **kwargs):
        return all([True and func(self, agent, proposed_state_change) for func in self.reqs])

class SpaceState(State):
    def __init__(self, region=None, dimension=2, agents=None, **kwargs):
        super().__init__(**kwargs)
        self.region = region
        self.dimension = dimension
        self.agents = agents

    def req_in_bounds(self, agent, proposed_position):
        """
        Return True if the proposed position is in bounds. The bounds is defined
        as [0, region)^N, where N is the spatial dimension.
        """
        return np.all(proposed_position >= 0) and np.all(proposed_position < self.region)

    def req_not_overlapped(self, agent, proposed_position):
        """
        Return True if the proposed position is not already occupied by another agent.
        """
        overlap = False
        for other in self.agents.values():
            if other.id == agent.id: continue # Cannot overlap with yourself
            if not isinstance(other, ContinuousSpaceAgent) or other.position is None: continue # Cannot overlap with agents that don't occupy space.
            if np.linalg.norm(other.position - proposed_position) < (other.radius + agent.radius):
                overlap = True
                break
        return not overlap
    
    
class ContinuousSpaceState(SpaceState):
    def __init__(self, random_reset_attempts=100, **kwargs):
        super().__init__(**kwargs)
        self.random_reset_attempts = random_reset_attempts
    
    def reset(self, **kwargs):
        # Reset all agent positions
        for agent in self.agents.values():
            if isinstance(agent, ContinuousSpaceAgent):
                agent.position = None
        
        # Set the positions for the agents with initial positions first. If these
        # agents cannot be placed in the space, then throw an error because the
        # environment designer has specified a bad setup.
        for agent in self.agents.values():
            if isinstance(agent, ContinuousSpaceAgent) and agent.initial_position is not None:
                assert len(agent.initial_position) == self.dimension, f"Mismatch in space dimension and {agent.id}'s initial position"
                if not self.set_position(agent, agent.initial_position, **kwargs):
                    raise ValueError(f"{agent.id}'s specified initial position is invalid in this space.")
        
        # Now set the position of any leftover agents that do not have initial
        # positions specified. Some attempts may fail, so attempt to set the agents
        # positions several times before giving up
        for agent in self.agents.values():
            if isinstance(agent, ContinuousSpaceAgent):
                could_not_place = True
                for _ in range(self.random_reset_attempts):
                    if self.set_position(agent, np.random.uniform(agent.radius, self.region - agent.radius, (self.dimension,))):
                        could_not_place = False
                        break
                if could_not_place:
                    raise RuntimeError(f"Could not randomly place the agents in the space with {self.random_reset_attempts} tries.")

    
    def set_position(self, agent, _position, **kwargs):
        if super().check_reqs(agent, _position):
            agent.position = _position
            return True
        return False
    
    def modify_position(self, agent, value, **kwargs):
        if isinstance(agent, ContinuousSpaceAgent):
            position_before = agent.position
            self.set_position(agent, agent.position + value, **kwargs)
            return agent.position - position_before

class StateWrapper:
    def __init__(self, wrapped_state, *args, **kwargs):
        self.wrapped_state = wrapped_state
    
    def reset(self, *args, **kwargs):
        return self.wrapped_state.reset(**kwargs)

    def set_position(self, *args, **kwargs):
        return self.wrapped_state.set_position(*args, **kwargs)
    
    def modify_position(self, *args, **kwargs):
        return self.wrapped_state.modify_position(*args, **kwargs)
    
    # TODO: would something like this work? Does it work work with an additional
    # property argument? How do we define the interface for the state components.
    # def set_property(self, *args, **kwargs):
    #     pass
        # self.wrapped_state.set_property(**kwargs)
    
    # def modify_property(self, *args, **kwargs):
    #     pass
        # self.wrapped_state.modify_property(**kwargs)

class GridSpaceWrapper(StateWrapper):
    def __init__(self, *args, grid=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert grid is not None, "Why would you use a grid wrapper without the grid?"
        if type(grid) is tuple:
            assert len(grid) == self.wrapped_state.dimension, "Grid dimension must match space dimension"
            self.grid = np.empty(grid, dtype=object) # Is object the right thing to use here for a list/dict of agents?
        elif type(grid) is int:
            self.grid = np.empty((grid,) * self.wrapped_state.dimension, dtype=object)
        else:
            raise TypeError("We only accept int or tuple for grid argument")




agents = {
    'agent0': ContinuousSpaceAgent(id='agent0', radius=1),
    'agent1': ContinuousSpaceAgent(id='agent1', radius=2),
    'agent2': ContinuousSpaceAgent(id='agent2', radius=0.5),
    'agent3': ContinuousSpaceAgent(id='agent3', radius=1),
    'agent4': ContinuousSpaceAgent(id='agent4', radius=1.2),
    'agent5': ContinuousSpaceAgent(id='agent5', radius=1.8),
    'agent6': ContinuousSpaceAgent(id='agent6', radius=0.76),
}

region = 10
state = ContinuousSpaceState(region=region, agents=agents, reqs=[ContinuousSpaceState.req_not_overlapped])
state = GridSpaceWrapper(state)
try:
    state.reset()
except RuntimeError:
    print('Runtime error')
    pass


from matplotlib import pyplot as plt
from admiral.tools.matplotlib_utils import mscatter
fig = plt.figure()
# Draw the resources
ax = fig.gca()

# Draw the agents
ax.set(xlim=(0, region), ylim=(0, region))
ax.set_xticks(np.arange(0, region, 1))
ax.set_yticks(np.arange(0, region, 1))

agents_x = [agent.position[0] for agent in agents.values() if agent.position is not None]
agents_y = [agent.position[1] for agent in agents.values() if agent.position is not None]
agents_size = [3000*agent.radius for agent in agents.values() if agent.position is not None]
mscatter(agents_x, agents_y, ax=ax, m='o', s=agents_size, edgecolor='black', facecolor='gray')

plt.grid()
plt.plot()
plt.pause(3)



