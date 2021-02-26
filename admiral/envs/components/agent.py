
from gym.spaces import Dict

# ------------------ #
# --- Base Agent --- #
# ------------------ #

class Agent:
    """
    Base Agent class for agents that live in an environment. Agents require an
    id in in order to even be constructed.
    """
    def __init__(self, id=None, **kwargs):
        if id is None:
            raise TypeError("Agents must be constructed with an id.")
        else:
            self.id = id
        
    def finalize(self, **kwargs):
        pass
    
    @property
    def configured(self):
        """
        Determine if the agent has been successfully configured.
        """
        return self.id is not None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

class ActingAgent(Agent):
    """
    ActingAgents are Agents that are expected to produce actions and therefore
    should have an action space in order to be successfully configured.
    """
    def __init__(self, action_space=None, **kwargs):
        super().__init__(**kwargs)
        self.action_space = {} if action_space is None else action_space

    def finalize(self, **kwargs):
        super().finalize(**kwargs)
        self.action_space = Dict(self.action_space)
    
    @property
    def configured(self):
        return super().configured and self.action_space

class ObservingAgent(Agent):
    """
    ObservingAgents are Agents that are expected to receive observations and therefore
    should have an observation space in order to be successfully configured.
    """
    def __init__(self, observation_space=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = {} if observation_space is None else observation_space

    def finalize(self, **kwargs):
        super().finalize(**kwargs)
        self.observation_space = Dict(self.observation_space)
    
    @property
    def configured(self):
        return super().configured and self.observation_space



# ----------------- #
# --- Attacking --- #
# ----------------- #

class AttackingAgent(ActingAgent):
    """
    Agents that can attack other agents.

    attack_range (int):
        The effective range of the attack. Can be used to determine if an attack
        is successful based on distance between agents.
    
    attack_strength (float):
        How effective the agent's attack is. This is applicable in situations where
        the agents' health is affected by attacks.
    """
    def __init__(self, attack_range=None, attack_strength=None, **kwargs):
        super().__init__(**kwargs)
        assert attack_range is not None, "attack_range must be a nonnegative integer"
        self.attack_range = attack_range
        assert attack_strength is not None, "attack_strength must be a nonnegative number"
        self.attack_strength = attack_strength
    
    @property
    def configured(self):
        """
        The agent is successfully configured if the attack range and strength is
        specified.
        """
        return super().configured and self.attack_range is not None and self.attack_strength is not None



# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class LifeAgent(Agent):
    """
    Agents have health and are alive or dead.

    min_health (float):
        The minimum value the health can reach before the agent dies.
    
    max_health (float):
        The maximum value the health can reach.

    initial_health (float):
        The initial health of the agent. The health will be set to this initial
        option at reset time.
    """
    def __init__(self, min_health=0.0, max_health=1.0, initial_health=None, **kwargs):
        super().__init__(**kwargs)
        if initial_health is not None:
            assert min_health <= initial_health <= max_health
        self.initial_health = initial_health
        self.min_health = min_health
        self.max_health = max_health
        self.is_alive = True
        self.health = None

    @property
    def configured(self):
        """
        The agent is successfully configured if the min and max health are specified
        and if is_alive is specified.
        """
        return super().configured and self.min_health is not None and self.max_health is not None and self.is_alive is not None

class LifeObservingAgent(ObservingAgent): pass
class HealthObservingAgent(ObservingAgent): pass


# ----------------- #
# --- Observing --- #
# ----------------- #

# TODO: move this to a more specific location
class AgentObservingAgent(ObservingAgent):
    """
    Agents can observe other agents.

    agent_view (int):
        Any agent within this many spaces will be fully observed.
    """
    def __init__(self, agent_view=None, **kwargs):
        """
        Agents can see other agents up to some maximal distance away, indicated
        by the view.
        """
        super().__init__(**kwargs)
        assert agent_view is not None, "agent_view must be nonnegative integer"
        self.agent_view = agent_view
    
    @property
    def configured(self):
        """
        Agents are configured if the agent_view parameter is set.
        """
        return super().configured and self.agent_view is not None



# ------------- #
# --- Plume --- #
# ------------- #

class PlumeSamplingAgent(ObservingAgent, ActingAgent):
    """
    Agents can sample the environment based on their location and receive a concentration
    measurement.

    noise (float):
        The magnitude of Gaussian noise to add to their concentration.
    """
    def __init__(self, noise=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise
    
    @property
    def configured(self):
        return super().configured and self.noise



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionAgent(Agent):
    """
    Agents have a position in the environment.

    initial_position (np.array):
        The desired starting position for this agent.

    Warning: You should consider the positional state of the environment when
    working with the agent's position because states interpret the elements of
    the array differently.
    """
    def __init__(self, initial_position=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_position = initial_position
        self.position = None

class PositionObservingAgent(ObservingAgent): pass

class GridMovementAgent(ActingAgent):
    """
    Agents can move up to some number of spaces away.

    move_range (int):
        The maximum number of cells away that the agent can move.
    """
    def __init__(self, move_range=None, **kwargs):
        super().__init__(**kwargs)
        assert move_range is not None, "move_range must be an integer"
        self.move_range = move_range
    
    @property
    def configured(self):
        """
        Agents are configured if the move_range parameter is set.
        """
        return super().configured and self.move_range is not None

class SpeedAngleAgent(Agent):
    """
    Agents have a speed and a banking angle which are used to determine how the
    agent moves around a continuous field.
    
    min_speed (float):
        The minimum speed this agent can travel.
    
    max_speed (float):
        The maximum speed this agent can travel.
    
    max_banking_angle (float):
        The maximum banking angle the agent can endure.

    initial_speed (float):
        The agent's initial speed.
    
    initial_banking_angle (float):
        The agent's initial banking angle.
    
    initial_ground_angle (float):
        The agent's initial ground angle.
    """
    def __init__(self, min_speed=0.25, max_speed=1.0, max_banking_angle=45, initial_speed=None, \
                 initial_banking_angle=None, initial_ground_angle=None, **kwargs):
        super().__init__(**kwargs)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.initial_speed = initial_speed
        self.speed = None # Should be set by the state handler

        self.max_banking_angle = max_banking_angle
        self.initial_banking_angle = initial_banking_angle
        self.initial_ground_angle = initial_ground_angle
        self.banking_angle = None # Should be set by the state handler
    
    @property
    def configured(self):
        return super().configured and self.min_speed is not None and self.max_speed is not None and \
            self.max_banking_angle is not None

class SpeedAngleActingAgent(ActingAgent):
    """
    Agents can change their speed and banking angles.
    
    max_acceleration (float):
        The maximum amount by which an agent can change its speed in a single time
        step.

    max_banking_angle_change (float):
        The maximum amount by which an agent can change its banking angle in a
        single time step.
    """
    def __init__(self, max_acceleration=0.25, max_banking_angle_change=30, **kwargs):
        super().__init__(**kwargs)
        self.max_acceleration = max_acceleration
        self.max_banking_angle_change = max_banking_angle_change
    
    @property
    def configured(self):
        return super().configured and self.max_acceleration is not None and \
            self.max_banking_angle_change is not None

class SpeedAngleObservingAgent(ObservingAgent): pass

class VelocityAgent(Agent):
    """
    Agents have a velocity which determines how it moves around in a continuous
    field. Agents can accelerate to modify their velocities.

    initial_velocity (np.array):
        Two-element float array that is the agent's initial velocity.

    max_speed (float):
        The maximum speed the agent can travel.
    
    max_acceleration (float):
        The maximum amount by which an agent can change its velocity in a single
        time step.
    """
    def __init__(self, initial_velocity=None, max_speed=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_velocity = initial_velocity
        self.max_speed = max_speed
    
    @property
    def configured(self):
        return super().configured and self.max_speed is not None

class AcceleratingAgent(ActingAgent):
    """
    Agents can accelerate to modify their velocities.
    
    max_acceleration (float):
        The maximum amount by which an agent can change its velocity in a single
        time step.
    """
    def __init__(self, max_acceleration=None, **kwargs):
        super().__init__(**kwargs)
        self.max_acceleration = max_acceleration
    
    @property
    def configured(self):
        return super().configured and self.max_acceleration is not None

class VelocityObservingAgent(ObservingAgent): pass

class CollisionAgent(Agent):
    """
    Agents that have physical size and mass and can be used in collisions.

    size (float):
        The size of the agent.
        Default 1.
    
    mass (float):
        The mass of the agent.
        Default 1.
    """
    def __init__(self, size=1, mass=1, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.mass = mass
    
    @property
    def configured(self):
        return super().configured and self.size is not None and self.mass is not None

# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class HarvestingAgent(ActingAgent):
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

class ResourceObservingAgent(ObservingAgent):
    """
    Agents can observe the resources in the environment.

    resource_view (int):
        Any resources within this range of the agent's position will be fully observed.
    """
    def __init__(self, resource_view=None, **kwargs):
        super().__init__(**kwargs)
        assert resource_view is not None, "resource_view must be nonnegative integer"
        self.resource_view = resource_view

    @property
    def configured(self):
        """
        Agents are configured if the resource_view parameter is set.
        """
        return super().configured and self.resource_view is not None



# ------------ #
# --- Team --- #
# ------------ #

class TeamAgent(Agent):
    """
    Agents are on a team, which will affect their ability to perform certain actions,
    such as who they can attack.
    """
    def __init__(self, team=None, **kwargs):
        super().__init__(**kwargs)
        assert team is not None, "team must be an integer"
        self.team = team
    
    @property
    def configured(self):
        """
        Agent is configured if team is set.
        """
        return super().configured and self.team is not None

class TeamObservingAgent(ObservingAgent): pass
