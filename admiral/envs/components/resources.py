
import numpy as np

from admiral.envs import Agent
from admiral.envs.components.position import PositionAgent

class ResourceObservingAgent(Agent):
    """
    Agents can observe the resources in the environment.

    resource_view_range (int):
        Any resources within this range of the agent's position will be fully observed.
    """
    def __init__(self, resource_view_range=None, **kwargs):
        super().__init__(**kwargs)
        assert resource_view_range is not None, "resource_view_range must be nonnegative integer"
        self.resource_view_range = resource_view_range
    
    @property
    def configured(self):
        """
        Agents are configured if the resource_view_range parameter is set.
        """
        return super().configured and self.resource_view_range is not None

class HarvestingAgent(Agent):
    """
    Agents can harvest resources.

    max_harvest (double):
        The maximum amount of resources the agent can harvest from the cell it
        occupies.
    """
    def __init__(self, max_harvest=None, **kwargs):
        super().__init__(**kwargs)
        assert max_harvest is not None, "max_harvest must be nonnegative number"
        self.max_harvest = max_harvest
    
    @property
    def configured(self):
        """
        Agents are configured if max_harvest is set.
        """
        return super().configured and self.max_harvest is not None

class GridResourceState:
    """
    Resources exist in the cells of the grid. The grid is populated with resources
    between the min and max value on some coverage of the region at reset time.
    If original resources is specified, then reset will set the resources back 
    to that original value. This component supports resource depletion: if a resource falls below
    the minimum value, it will not regrow. Agents can harvest resources from the cell they occupy.
    Agents can observe the resources in a grid-like observation surrounding their positions.

    The action space of GridResourcesHarvestingAgents is appended with
    Box(0, agent.max_harvest, (1,), np.float), indicating that the agent can harvest
    up to its max harvest value on the cell it occupies.

    The observation space of ObservingAgents is appended with
    Box(0, self.max_value, (agent.view*2+1, agent.view*2+1), np.float), indicating
    that an agent can observe the resources in a grid surrounding its position,
    up to its view distance.

    agents (dict):
        The dictionary of agents. Because agents harvest and observe resources
        based on their positions, agents must be GridPositionAgents.

    region (int):
        The size of the region

    coverage (float):
        The ratio of the region that should start with resources.

    min_value (float):
        The minimum value a resource can have before it cannot grow back. This is
        different from the absolute minimum value, 0, which indicates that there
        are no resources in the cell.
    
    max_value (float):
        The maximum value a resource can have.

    regrow_rate (float):
        The rate at which resources regrow.
    
    original_resources (np.array):
        Instead of specifying the above resource-related parameters, we can provide
        an initial state of the resources. At reset time, the resources will be
        set to these original resources. Otherwise, the resources will be set
        to random values between the min and max value up to some coverage of the
        region.
    """
    def __init__(self, agents=None, region=None, coverage=0.75, min_value=0.1, max_value=1.0,
            regrow_rate=0.04, original_resources=None, **kwargs):        
        self.original_resources = original_resources
        if self.original_resources is None:
            assert type(region) is int, "Region must be an integer."
            self.region = region
        else:
            self.region = self.original_resources.shape[0]
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage

        assert type(agents) is dict, "agents must be a dict"
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the resources. If original resources is specified, then the resources
        will be reset back to this original value. Otherwise, the resources will
        be randomly generated values between the min and max value up to some coverage
        of the region.
        """
        if self.original_resources is not None:
            self.resources = self.original_resources
        else:
            coverage_filter = np.zeros((self.region, self.region))
            coverage_filter[np.random.uniform(0, 1, (self.region, self.region)) < self.coverage] = 1.
            self.resources = np.multiply(
                np.random.uniform(self.min_value, self.max_value, (self.region, self.region)),
                coverage_filter
            )
    
    def set_resources(self, location, value, **kwargs):
        """
        Set the resource at a certain location to a value, bounded between 0 and
        the maximum resource value.
        """
        assert type(location) is tuple
        if value <= 0:
            self.resources[location] = 0
        elif value >= self.max_value:
            self.resources[location] = self.max_value
        else:
            self.resources[location] = value
    
    def modify_resources(self, location, value, **kwargs):
        """
        Add some value to the resource at a certain location.
        """
        assert type(location) is tuple
        self.set_resources(location, self.resources[location] + value, **kwargs)

    def regrow(self, **kwargs):
        """
        Regrow the resources according to the regrow_rate.
        """
        self.resources[self.resources >= self.min_value] += self.regrow_rate
        self.resources[self.resources >= self.max_value] = self.max_value

class GridResourcesActor:
    """
    Provides the necessary action space for agents who can harvest resources and
    processes the harvesting action.

    resources (ResourceState):
        The resource state handler.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, HarvestingAgent):
                agent.action_space['harvest'] = Box(0, agent.max_harvest, (1,), np.float)

    def process_harvest(self, agent, amount, **kwargs):
        """
        Harvest some amount of resources at the agent's position.

        agent (HarvestingAgent):
            The agent who has chosen to harvest the resource.

        amount (float):
            The amount of resource the agent wants to harvest.
        
        return (float):
            Return the amount of resources that was actually harvested. This can
            be less than the desired amount if the cell does not have enough resources.
        """
        location = tuple(agent.position)
        resource_before = self.resources.resources[location]
        self.resources.modify_resources(location, -amount)
        return resource_before - self.resources.resources[location]

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view_range centered on their
    position. The values in the grid are the values of the resources in that
    area.

    resources (ResourceState):
        The resource state handler.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, resources=None, agents=None, **kwargs):
        self.resources = resources
        self.agents = agents

        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, ResourceObservingAgent):
                agent.observation_space['resources'] = Box(0, self.resources.max_value, (agent.resource_view_range*2+1, agent.resource_view_range*2+1), np.float)

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view_range*2+1, agent.resource_view_range*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position
            r_lower = max([0, r-agent.resource_view_range])
            r_upper = min([self.resources.region-1, r+agent.resource_view_range])+1
            c_lower = max([0, c-agent.resource_view_range])
            c_upper = min([self.resources.region-1, c+agent.resource_view_range])+1
            signal[(r_lower+agent.resource_view_range-r):(r_upper+agent.resource_view_range-r),(c_lower+agent.resource_view_range-c):(c_upper+agent.resource_view_range-c)] = \
                self.resources.resources[r_lower:r_upper, c_lower:c_upper]
            return signal