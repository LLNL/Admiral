
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

from admiral.envs.components.agent import LifeAgent, PositionAgent, SpeedAngleAgent, VelocityAgent, \
    CollisionAgent

# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeState:
    """
    Agents can die if their health falls below their minimal health value. Health
    can decrease in a number of interactions. This environment provides an entropy
    that indicates how much health an agent loses when apply_entropy is called.
    This is a generic entropy for the step. If you want to specify health changes
    for specific actions, such as being attacked or harvesting, you must write
    it in the environment.

    agents (dict):
        Dictionary of agents.
    
    entropy (float):
        The amount of health that is depleted from an agent whenever apply_entropy
        is called.
    """
    def __init__(self, agents=None, entropy=0.1, **kwargs):
        assert type(agents) is dict, "Agents must be a dict"
        self.agents = agents
        self.entropy = entropy
    
    def reset(self, **kwargs):
        """
        Reset the health and life state of all applicable agents.
        """
        for agent in self.agents.values():
            if isinstance(agent, LifeAgent):
                if agent.initial_health is not None:
                    agent.health = agent.initial_health
                else:
                    agent.health = np.random.uniform(agent.min_health, agent.max_health)
                agent.is_alive = True
    
    def set_health(self, agent, _health):
        """
        Set the health of an agent to a specific value, bounded by the agent's
        min and max health-value. If that value is less than the agent's health,
        then the agent dies.
        """
        if isinstance(agent, LifeAgent):
            if _health <= agent.min_health:
                agent.health = 0
                agent.is_alive = False
            elif _health >= agent.max_health:
                agent.health = agent.max_health
            else:
                agent.health = _health
    
    def modify_health(self, agent, value):
        """
        Add some value to the health of the agent.
        """
        if isinstance(agent, LifeAgent):
            self.set_health(agent, agent.health + value)

    def apply_entropy(self, agent, **kwargs):
        """
        Apply entropy to the agent, decreasing its health by a small amount.
        """
        if isinstance(agent, LifeAgent):
            self.modify_health(agent, -self.entropy, **kwargs)



# ------------- #
# --- Plume --- #
# ------------- #

class PlumeState:
    """
    The source of a plume exists somewhere within the region. It has a strength
    and Gaussian diffusion. There is also wind, which blows the plume a certain
    direction. This disperses concentrations through the region, which the agents
    can try to measure.

    The plume is static throughout the entire episode.

    region (int):
        The size of the region.

    initial_source_location (np.array):
        The (x, y, z) location of the plume's source. If None, then the source
        will appear in a different location at the start of every episode.
    
    strength (float):
        The strength of the plume source, which impacts the concentration across
        the region.
    
    initial_wind_velocity (np.array):
        The velocity of the wind.
    
    TODO: Not entirely sure what these two do. I think the y-direction disk will
          have to update to be in the direction that is perpendicular to the wind
          velocity.
    disk_area_y (float):
        Disk area parameter in the y direction.
    
    disk_area_z (float):
        Disk area parameter in the z direction.
    """
    def __init__(self, region= None, initial_source_location=None, strength=10., initial_wind_velocity=None, \
        disk_area_y=0.5, disk_area_z=0.2):
        self.region = region
        self.strength = strength
        self.disk_area_y = disk_area_y
        self.disk_area_z = disk_area_z

        self.initial_source_location = initial_source_location
        self.initial_wind_velocity = initial_wind_velocity

        # These will be set at reset
        self.source_location = None
        self.wind_velocity = None
    
    def reset(self, **kwargs):
        """
        Reset the plume. If the plume has initial source and wind velocity, then
        use those. Otherwise, randomly pick a location within the region for the
        source and a random wind velocity between 1 and 10 1-norm.
        """
        if self.initial_source_location is not None:
            self.source_location = self.initial_source_location
        else:
            self.source_location = np.random.uniform(0, self.region, 2)
        
        if self.initial_wind_velocity is not None:
            self.wind_velocity = self.initial_wind_velocity
        else:
            self.wind_velocity = np.random.uniform(1, 10, 2)
    
        # Add all the extra wind parameters needed for observing the concentrations
        self.wind_speed = np.linalg.norm(self.wind_velocity)
        self.wind_unit_vector = self.wind_velocity / self.wind_speed
        self.wind_perpendicular = np.array([self.wind_velocity[1], -self.wind_velocity[0]])
        radian = np.arccos(np.dot(np.array([1., 0.]), self.wind_unit_vector))
        self.wind_angle = radian if self.wind_velocity[1] >= 0. else -radian
        self.wind_rotation_matrix = sp.linalg.expm(np.cross(np.eye(2), -self.wind_angle * np.array([0., 1.])))


# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionState(ABC):
    """
    Manages the agents' positions.

    region (int):
        The size of the environment.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, region=None, agents=None, **kwargs):
        assert type(region) is int, "Region must be an integer."
        self.region = region
        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the agents' positions. If the agents were created with a starting
        position, then use that. Otherwise, randomly assign a position in the region.
        """
        # Invalidate all the agents' positions from last episode
        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                agent.position = None

        for agent in self.agents.values():
            if isinstance(agent, PositionAgent):
                if agent.initial_position is not None:
                    agent.position = agent.initial_position
                else:
                    self.random_reset(agent)
    
    @abstractmethod
    def random_reset(self, agent, **kwargs):
        """
        Reset the agents' positions. Child classes implement this according to their
        specs. For example, GridPositionState assigns random integers as the position,
        whereas ContinuousPositionState assigns random numbers.
        """
        pass

    @abstractmethod
    def set_position(self, agent, position, **kwargs):
        """
        Set the position of the agents. Child classes implement.
        """
        pass
    
    def modify_position(self, agent, value, **kwargs):
        """
        Add some value to the position of the agent.
        """
        if isinstance(agent, PositionAgent):
            self.set_position(agent, agent.position + value)

class GridPositionState(PositionState):
    """
    Agents are positioned in a grid and cannot go outside of the region. Positions
    are a 2-element numpy array, where the first element is the grid-row from top
    to bottom and the second is the grid-column from left to right.
    """
    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value only if the new position
        is within the region.
        """
        if isinstance(agent, PositionAgent):
            if 0 <= _position[0] < self.region and 0 <= _position[1] < self.region:
                agent.position = _position

    def random_reset(self, agent, **kwargs):
        """
        Set the agents' random positions as integers within the region.
        """
        agent.position = np.random.randint(0, self.region, 2)

class ContinuousPositionState(PositionState):
    """
    Agents are positioned in a continuous space and can go outside the bounds
    of the region. Positions are a 2-element array, where the first element is
    the x-location and the second is the y-location.
    """
    def __init__(self, reset_attempts=100, **kwargs):
        super().__init__(**kwargs)
        self.reset_attempts = reset_attempts

    def set_position(self, agent, _position, **kwargs):
        """
        Set the agent's position to the incoming value.
        """
        if isinstance(agent, PositionAgent):
            agent.position = _position

    def random_reset(self, agent, **kwargs):
        """
        Set the agents' random positions as numbers within the region.
        """
        if isinstance(agent, CollisionAgent):
            for _ in range(self.reset_attempts):
                potential_position = np.random.uniform(0, self.region, 2)
                collision = False
                for other in self.agents.values():
                    if other.id != agent.id and \
                       isinstance(other, CollisionAgent) and \
                       other.position is not None and \
                       np.linalg.norm(other.position - potential_position) < (other.size + agent.size):
                        collision = True
                        break
                if not collision:
                    agent.position = potential_position
                    return
            raise Exception("Could not fit all the agents in the region without collisions")
        else:
            agent.position = np.random.uniform(0, self.region, 2)

class SpeedAngleState:
    """
    Manages the agents' speed, banking angles, and ground angles.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
    
    def reset(self, **kwargs):
        """
        Reset the agents' speeds and ground angles.
        """
        for agent in self.agents.values():
            if isinstance(agent, SpeedAngleAgent):
                # Reset agent speed
                if agent.initial_speed is not None:
                    agent.speed = agent.initial_speed
                else:
                    agent.speed = np.random.uniform(agent.min_speed, agent.max_speed)

                # Reset agent banking angle
                if agent.initial_banking_angle is not None:
                    agent.banking_angle = agent.initial_banking_angle
                else:
                    agent.banking_angle = np.random.uniform(-agent.max_banking_angle, agent.max_banking_angle)

                # Reset agent ground angle
                if agent.initial_ground_angle is not None:
                    agent.ground_angle = agent.initial_ground_angle
                else:
                    agent.ground_angle = np.random.uniform(0, 360)
    
    def set_speed(self, agent, _speed, **kwargs):
        """
        Set the agent's speed if it is between its min and max speed.
        """
        if isinstance(agent, SpeedAngleAgent):
            if agent.min_speed <= _speed <= agent.max_speed:
                agent.speed = _speed
    
    def modify_speed(self, agent, value, **kwargs):
        """
        Modify the agent's speed.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_speed(agent, agent.speed + value)
    
    def set_banking_angle(self, agent, _banking_angle, **kwargs):
        """
        Set the agent's banking angle if it is between its min and max angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            if abs(_banking_angle) <= agent.max_banking_angle:
                agent.banking_angle = _banking_angle
                self.modify_ground_angle(agent, agent.banking_angle)
    
    def modify_banking_angle(self, agent, value, **kwargs):
        """
        Modify the agent's banking angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_banking_angle(agent, agent.banking_angle + value)

    def set_ground_angle(self, agent, _ground_angle, **kwargs):
        """
        Set the agent's ground angle, which will be modded to fall between 0 and
        360.
        """
        if isinstance(agent, SpeedAngleAgent):
            agent.ground_angle = _ground_angle % 360
    
    def modify_ground_angle(self, agent, value, **kwargs):
        """
        Modify the agent's ground angle.
        """
        if isinstance(agent, SpeedAngleAgent):
            self.set_ground_angle(agent, agent.ground_angle + value)

class VelocityState:
    """
    Manages the agents' velocities.
    """
    def __init__(self, agents=None, friction=0.05, **kwargs):
        self.agents = agents
        self.friction = friction
    
    def reset(self, **kwargs):
        """
        Reset the agents' velocities.
        """
        for agent in self.agents.values():
            if isinstance(agent, VelocityAgent):
                # Reset the agent's velocity
                if agent.initial_velocity is not None:
                    agent.velocity = agent.initial_velocity
                else:
                    agent.velocity = np.random.uniform(-agent.max_speed, agent.max_speed, (2,))
    
    def set_velocity(self, agent, _velocity, **kwargs):
        """
        Set the agent's velocity if it is within its max speed.
        """
        if isinstance(agent, VelocityAgent):
            vel_norm = np.linalg.norm(_velocity)
            if vel_norm < agent.max_speed:
                agent.velocity = _velocity
            else:
                agent.velocity = _velocity / vel_norm * agent.max_speed
    
    def modify_velocity(self, agent, value, **kwargs):
        """
        Modify the agent's velocity.
        """
        if isinstance(agent, VelocityAgent):
            self.set_velocity(agent, agent.velocity + value, **kwargs)
    
    def apply_friction(self, agent, **kwargs):
        """
        Apply friction to the agent's movement, decreasing its speed by a small amount.
        """
        if isinstance(agent, VelocityAgent):
            old_speed = np.linalg.norm(agent.velocity)
            new_speed = old_speed - self.friction
            if new_speed <= 0:
                agent.velocity = np.zeros(2)
            else:
                agent.velocity *= new_speed / old_speed



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceState:
    """
    Resources exist in the cells of the grid. The grid is populated with resources
    between the min and max value on some coverage of the region at reset time.
    If original resources is specified, then reset will set the resources back 
    to that original value. This component supports resource depletion: if a resource falls below
    the minimum value, it will not regrow. Agents can harvest resources from the cell they occupy.
    Agents can observe the resources in a grid-like observation surrounding their positions.

    An agent can harvest up to its max harvest value on the cell it occupies. It
    can observe the resources in a grid surrounding its position, up to its view
    distance.

    agents (dict):
        The dictionary of agents.

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
    
    initial_resources (np.array):
        Instead of specifying the above resource-related parameters, we can provide
        an initial state of the resources. At reset time, the resources will be
        set to these original resources. Otherwise, the resources will be set
        to random values between the min and max value up to some coverage of the
        region.
    """
    def __init__(self, agents=None, region=None, coverage=0.75, min_value=0.1, max_value=1.0,
            regrow_rate=0.04, initial_resources=None, **kwargs):        
        self.initial_resources = initial_resources
        if self.initial_resources is None:
            assert type(region) is int, "Region must be an integer."
            self.region = region
        else:
            self.region = self.initial_resources.shape[0]
        self.min_value = min_value
        self.max_value = max_value
        self.regrow_rate = regrow_rate
        self.coverage = coverage

        assert type(agents) is dict, "agents must be a dict"
        self.agents = agents

    def reset(self, **kwargs):
        """
        Reset the resources. If original resources is specified, then the resources
        will be reset back to this original value. Otherwise, the resources will
        be randomly generated values between the min and max value up to some coverage
        of the region.
        """
        if self.initial_resources is not None:
            self.resources = self.initial_resources
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



# ------------ #
# --- Team --- #
# ------------ #

class TeamState:
    """
    Team state manages the state of agents' teams. Since these are not changing,
    there is not much to manage. It really just keeps track of the number_of_teams.

    number_of_teams (int):
        The number of teams in this simulation.
    """
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        self.number_of_teams = number_of_teams
        self.agents = agents
