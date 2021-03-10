
from abc import ABC, abstractmethod

import numpy as np

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



class PlumeModel(object):
    """Simple concentration model with constant Gaussian emission."""

    ORIGIN = np.array([0., 0., 0.])
    BASE_SYSTEM = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])

    def __init__(self, params=None):
        """Simple plume model with Gaussian emission. The plume is
        characterized by the location of the source, the strength of the
        source, the wind-speed in the x-direction, the disk-area in the
        y-direction, the disk-area in the z-direction, and the magnitude
        of noise to add to the samples.

        Parameters:
            source_coords:        (x, y, z) tuple indicating the location and
                                  height of the plume
            strength:             source strength in grams / second
            wind_speed:           wind speed along x direction in meters/second
            diffusion_factor_y:   diffusion factor in y direction for urban conditions
            diffusion_factor_z:   disk area parameter on z direction
            noise:              variance to add Gaussian noise to observation
            rng:                random number generator for reproducibility
        """
        self.params = params
        # get range of log values
        self.params['log_range'] = np.log(self.params['upper_bound']) - np.log(self.params['lower_bound'])
        # Initilize the discrete coordinate grid used for imaging and inference
        mesh = {}
        # prepare grid
        mesh['x_range'] = np.arange(-1.0, 1.0 + self.params['mesh_distance'], self.params['mesh_distance'])
        mesh['y_range'] = np.arange(-1.0, 1.0 + self.params['mesh_distance'], self.params['mesh_distance'])
        mesh['concentrations'] = None
        self.params['mesh_grid'] = mesh

        # self.params['plume_visual'] = None
        self.reset(first=True)

    def reset(self, source=[0., 0., 1.], wind_velocity=[1., 0., 0.], first=False):
        """ Resets the plume model and loads basic parameters depending
        on wind direction and speed. """
        # Set wind parameter
        wind_velocity = np.array(wind_velocity)
        self.params['wind'] = self.get_wind_params(wind_velocity)
        self.params['inverse_wind'] = self.get_wind_params(-1. * wind_velocity)

        self.params['source'] = np.array(source)
        if self.params['source'].shape == (2,):
            temp = np.array([0., 0., 1.])
            temp[0:2] = self.params['source'][0:2]
            self.params['source'] = temp
        if not first:
            self.params['mesh_grid'] = self._get_meshgrid()
            # self.params['plume_visual'] = self._get_plume_visual()

    def get_wind_params(self, wind_velocity):
        params = {}
        params['velocity'] = np.array(wind_velocity)
        params['speed'] = np.linalg.norm(wind_velocity)
        params['unit_vector'] = self._get_unit_vector(wind_velocity)
        params['perpendicular_vector'] = self._get_perpendicular_vector(params['unit_vector'])
        params['angle'] = self._get_angle(params['unit_vector'])
        params['rotation_matrix'] = self._get_rotation_matrix(
            self.BASE_SYSTEM[2], -params['angle'])
        return params

    def get_inverse_concentration_mesh(self, source_coords):
        '''
        This function privides concentrations for a mesh grid of discrete
        coordinates, similar to _get_meshgrid, however it does so for a plume
        with inverted wind direction. By inputing the sensor_coords for the
        source_coords of this model we are able to get model predictions of the
        concentration at the sensor location for hypothetical source locations
        on the meshgrid. These concentrations can then be used to estimate the
        likelihood of the source being at that meshgrid location for a sensor
        concentration measurement at source_coords.
        TODO: improve paramter notation so it is less confusion, perhaps at the cost of it being less generic.
        '''
        return self._get_meshgrid(source=source_coords, wind=self.params['inverse_wind'])

    def get_centered_mesh(self):
        return self._get_meshgrid(source=np.array((0., 0.)), wind=self.params['wind'])

    def get_concentration(self,
                          sensor_coords,
                          source_coords=None,
                          wind=None):
        """Generate concentration at given sensor location in micrograms per
        cubic meter.

        sensor_coords:   Numpy array where each row has the coords of a sensor
                         and a shape of (x,3)
        source_coords:   Use instead of actual source_coords when
                         calculating estimation error, shape (3,)
        """
        # print('input sensor_coords')
        # print(sensor_coords)
        # print('input source_coords')
        # print(source_coords)

        if wind is None:
            wind = self.params['wind']
        # Reshape sensor coordinates if necessary
        if sensor_coords.shape == (3,):
            sensor_coords = sensor_coords[np.newaxis, :]

        # Get source coords if hard-coded
        if source_coords is None:
            source_coords = self.params['source']
        source_coords = self._transform_coords(source_coords, sensor_count=len(sensor_coords))
        sensor_coords = self._transform_coords(sensor_coords)

        # Get relative coords of sensors to plume source
        # print('modified sensor_coords')
        # print(sensor_coords)
        # print('modified source_coords')
        # print(source_coords)
        rel_sensor_coords = sensor_coords - source_coords

        # Wind transformations
        # print('source_coords')
        # print(source_coords)
        # print("wind['perpendicular_vector']")
        # print(wind['perpendicular_vector'])
 
        # print('source_coords[:, 0:2]')
        # print(source_coords[:, 0:2])
        # print("wind['perpendicular_vector'][0:2]")
        # print(wind['perpendicular_vector'][0:2])
        line_point = source_coords[:, 0:2] + wind['perpendicular_vector'][0:2]
        wind_barrier = np.cross(
            rel_sensor_coords[:, 0:2],
            line_point - sensor_coords[:, 0:2])

        # Transform sensor coordinates to fit with wind direction
        if not np.array_equal(wind['unit_vector'], self.BASE_SYSTEM[0]):
            rel_sensor_coords = self._rotate_coords(
                wind['rotation_matrix'], rel_sensor_coords, source_coords[0])

        # Calculate concentration using a specific concentration model
        concentration = self._use_concentration_model(
            rel_sensor_coords,
            sensor_coords,
            source_coords)

        # Add noise if model has noise
        if self.params['has_noise']:
            concentration += self.params['rng'].normal(0, np.sqrt(self.params['noise']))

        # clip results for for numerical stability
        concentration = np.clip(concentration, self.params['lower_bound'], self.params['upper_bound'])
        # sensors "behind" the plume do not get any signal
        # Avoid zero values to not mess up log calculations
        concentration = np.where(
            wind_barrier > 0,
            concentration,
            self.params['offset'])

        # Apply sensitivity factor of sensors
        concentration *= self.params['sensibility']

        # Bring measurements in range [0,1]
        if self.params['normalize']:
            # bring back to sensible values
            concentration[concentration <= self.params['lower_bound']] = self.params['lower_bound']
            concentration[concentration > self.params['upper_bound']] = self.params['upper_bound']
            # take log to have a nicer range
            concentration = np.log(concentration)
            # move to positive values
            concentration -= np.log(self.params['lower_bound'])
            # Normalize to [0,1] using max log range
            concentration /= self.params['log_range']

        return concentration

    # def get_visual(self):
    #     return self.params['plume_visual']

    def gradient(self, sensor_coords, source_coords, observation):
        """
            Returns the gradient w.r.t. x_0, y_0, and z_0 for a given location source_coords
        """
        return self.grad_lhood(sensor_coords, source_coords, observation)

    def _get_meshgrid(self, source=None, wind=None):
        if source is None:
            source = self.params['source']
        mesh = self.params['mesh_grid']
        coords = np.array(np.meshgrid(mesh['x_range'], mesh['y_range'], [1.])).T.reshape(-1,3)
        # Get concentrations
        concentrations = self.get_concentration(
            sensor_coords=coords,
            source_coords=source,
            wind=wind)
        # Bring concentrations in right form for plot
        mesh['concentrations'] = concentrations.reshape(mesh['x_range'].shape[0], mesh['y_range'].shape[0]).T
        return mesh

    # def _get_plume_visual(self):
    #     # Plot concentration as mesh
    #     fig, ax = plt.subplots(frameon=False, figsize=(700, 700), dpi=1)
    #     fig.subplots_adjust(0,0,1,1)
    #     cs = ax.contourf(
    #         self.params['mesh_grid']['x_range'],
    #         self.params['mesh_grid']['y_range'],
    #         self.params['mesh_grid']['concentrations'], cmap=matplotlib.cm.OrRd) #cmap='YlOrRd') # levels=10)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     fig.canvas.draw()
    #     buffer = fig.canvas.tostring_rgb()
    #     buffer_array = np.fromstring(buffer, dtype=np.uint8, sep='')
    #     visual = buffer_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     plt.close('all')
    #     return visual

    def _transform_coords(self, coords, sensor_count=None):
        """Transform coordinates from 2D to 3D and extend if necessary."""
        # Properly transform coords into matrix
        # From [x,y,z] to [[x,y,z],[x,y,z],...,[x,y,z]]
        if not sensor_count is None:
            coords = np.repeat(
                np.expand_dims(coords, axis=0),
                [sensor_count],
                axis=0)
        # Properly transform sensor coords from 2D to 3D if necessary
        # From [[...,...],[...,...],...] to [[...,...,1.0],[...,...,1.0],...]
        # From [...,...] to [...,...,1.0]
        if coords.shape[1] == 2:
            coords = np.hstack((coords, np.ones((coords.shape[0],1))))
        return coords

    def _get_unit_vector(self, vector):
        """ Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def _get_perpendicular_vector(self, vector):
        """ Returns a vector that is perpenticular to a given vector
        in the z plane.
        """
        return np.cross(self.BASE_SYSTEM[2], vector, axis=0)

    def _get_angle(self, vector, base_vector=None):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        if base_vector is None:
            base_vector = self.BASE_SYSTEM[0]
        base_unit_vector = self._get_unit_vector(base_vector)
        unit_vector = self._get_unit_vector(vector)
        radian = np.arccos(np.clip(np.dot(base_unit_vector, unit_vector), -1.0, 1.0))
        if vector[1] >= 0:
            return radian
        return -radian

    def _get_rotation_matrix(self, axis, theta):
        """ Returns the Euler-Rodrigues rotation matrix. """
        from scipy import linalg
        return linalg.expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * theta))

    def _rotate_coords(self, rotation_matrix, coords, source=None):
        """ Rotate coordinates according to the Euler-Rodrigues rotation matrix. """
        if source is None:
            source = self.ORIGIN
        if coords.shape == (3,):
            return np.dot(rotation_matrix, coords) #+ source
        new = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            new[i] = self._rotate_coords(rotation_matrix, coords[i]) #+ source
        return new

    def __str__(self):
        indent = max([len(key) for key in [*self.params]]) + 5
        text  = 'PlumeModel_v0:\n'
        for key, value in self.params.items():
            text += '    {:_<{}s} {}\n'.format(key, indent, value)
        return text

    def __repr__(self):
        return self.__str__()

    def _use_concentration_model(self, rel_sensor_coords, sensor_coords, source_coords):
        # Calculate dispersion parameters sigma_y and sigma_z
        # as functions of downwind distance and stability
        sigma_y = \
            self.params['diffusion_factor_y'] *\
            rel_sensor_coords[:,0] *\
            (1 + 0.0004 * rel_sensor_coords[:,0])**(-0.5)
        sigma_z = \
            self.params['diffusion_factor_z'] *\
            rel_sensor_coords[:,0]
        # Calculate concentration factors
        a_t = -0.5 * (rel_sensor_coords[:,2] / sigma_z)**2
        b_t = -0.5 * ((sensor_coords[:,2] + source_coords[:,2]) / sigma_z)**2
        c_t = -0.5 * (rel_sensor_coords[:,1] / sigma_y)**2
        # Put everything together
        return \
            (self.params['strength'] / (2 * np.pi * sigma_y * sigma_z * self.params['wind']['speed'])) *\
            np.exp(c_t) *\
            (np.exp(a_t) + np.exp(b_t))

    def _use_concentration_model(self, rel_sensor_coords, sensor_coords, source_coords):
        # Calculate dispersion parameters sigma_y and sigma_z
        # as functions of downwind distance and stability
        sigma_y = \
            self.params['diffusion_factor_y'] *\
            rel_sensor_coords[:,0] *\
            (1 + 0.0004 * rel_sensor_coords[:,0])**(-0.5)
        sigma_z = \
            self.params['diffusion_factor_z'] *\
            rel_sensor_coords[:,0]
        # Calculate concentration factors
        a_t = -0.5 * (rel_sensor_coords[:,2] / sigma_z)**2
        b_t = -0.5 * ((sensor_coords[:,2] + source_coords[:,2]) / sigma_z)**2
        c_t = -0.5 * (rel_sensor_coords[:,1] / sigma_y)**2
        # Put everything together
        return \
            (self.params['strength'] / (2 * np.pi * sigma_y * sigma_z * self.params['wind']['speed'])) *\
            np.exp(c_t) *\
            (np.exp(a_t) + np.exp(b_t))





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
