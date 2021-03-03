
from gym.spaces import Box, MultiBinary, Dict
import numpy as np

from admiral.envs.components.agent import HealthObservingAgent, LifeObservingAgent, \
    AgentObservingAgent, PositionObservingAgent, SpeedAngleObservingAgent, VelocityObservingAgent, \
    ResourceObservingAgent, TeamObservingAgent, PositionAgent, LifeAgent, TeamAgent, \
    SpeedAngleAgent, VelocityAgent

# ----------------- #
# --- Utilities --- #
# ----------------- #

def obs_filter(obs, agent, agents, null_value):
    """
    Modify the observation, inserting null values for observation of agents that
    are too far way from the observing agent.
    """
    if isinstance(agent, PositionAgent) and \
       isinstance(agent, AgentObservingAgent):
        for other in agents.values():
            if other.id not in obs: continue # This agent is not observed
            if other.id == agent.id: continue # Don't modify yourself
            if not isinstance(other, PositionAgent):
                # Agents without positions will not be observed at all
                # TODO: make this a parameter to the observers below
                obs[other.id] = null_value
            else:
                pos_diff = abs(other.position - agent.position)
                if np.any(pos_diff > agent.agent_view): # Agent too far away to observe
                    obs[other.id] = null_value
    return obs



# ----------------------- #
# --- Health and Life --- #
# ----------------------- #

class HealthObserver:
    """
    Observe the health state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, HealthObservingAgent):
                agent.observation_space['health'] = Dict({
                    other.id: Box(-1, other.max_health, (1,), np.float) for other in self.agents.values() if isinstance(other, LifeAgent)
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the health state of all the agents in the simulator.

        agent (HealthObservingAgent):
            The agent making the observation.
        """
        if isinstance(agent, HealthObservingAgent):
            return {'health': {agent.id: agent.health for agent in self.agents.values() if isinstance(agent, LifeAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return -1

class PositionRestrictedHealthObserver(HealthObserver):
    """
    Observe the health of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}


class LifeObserver:
    """
    Observe the life state of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents

        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, LifeObservingAgent):
                agent.observation_space['life'] = Dict({
                    other.id: Box(-1, 1, (1,), np.int) for other in self.agents.values() if isinstance(other, LifeAgent)
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the life state of all the agents in the simulator.

        agent (LifeObservingAgent):
            The agent making the observation.
        """
        if isinstance(agent, LifeObservingAgent):
            return {'life': {agent.id: agent.is_alive for agent in self.agents.values() if isinstance(agent, LifeAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return -1

class PositionRestrictedLifeObserver(LifeObserver):
    """
    Observe the life of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}



# --------------- #
# --- Masking --- #
# --------------- #

class MaskObserver:
    """
    Observe a mask of each agent in the simulator. Agents that can be seen are
    indicatead with True, agents that cannot be seen (e.g. they are too far away)
    are indicated with False.
    """
    # TODO: Having a single mask observer may actually be confusing because it will
    # have every single agent in the dictionary, whereas the actual observer may
    # only have a subset of them. For example, a life observer will only include
    # life agents, and if this mask is applied, then the mask will include all
    # the agents, so there is not a clear mapping to the actual observation.
    #
    # Paths to take:
    # (1) Change the observers so that all agents are always observed
    # and the ones that are not life agents will simply be observed with the null
    # value.
    # (2) Have a separate mask for every observer as part of the restricted classes
    # below.
    # We should look at some rllib examples to help us decide the right way forward.
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent):
                agent.observation_space['mask'] = Dict({
                    other: MultiBinary(1) for other in agents
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the masking of all the agents in the simulator.
        """
        if isinstance(agent, AgentObservingAgent):
            return {'mask': {other: True for other in self.agents}}
        else:
            return {}
    
    @property
    def null_value(self):
        return False

class PositionRestrictedMaskObserver(MaskObserver):
    """
    Set the mask value to False for any agent that is too far way from the observing
    agent.
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}



# ----------------------------- #
# --- Position and Movement --- #
# ----------------------------- #

class PositionObserver:
    """
    Observe the positions of all the agents in the simulator.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Dict({
                    other.id: Box(-1, self.position.region, (2,), np.int) for other in agents.values() if isinstance(other, PositionAgent)
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionObservingAgent):
            return {'position': {other.id: other.position for other in self.agents.values() if isinstance(other, PositionAgent)}}
        else:
            return {}   
    
    @property
    def null_value(self):
        return np.array([-1, -1])

class PositionRestrictedPositionObserver(PositionObserver):
    """
    Observe the position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}


class RelativePositionObserver:
    """
    Observe the relative positions of agents in the simulator.
    """
    def __init__(self, position_state=None, agents=None, **kwargs):
        self.position_state = position_state
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, PositionObservingAgent) and \
               isinstance(agent, PositionAgent):
                agent.observation_space['position'] = Dict({
                    other.id: Box(-self.position_state.region, self.position_state.region, (self.position_state.dimension,), np.int) for other in agents.values() if (other.id != agent.id and isinstance(other, PositionAgent))
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the relative positions of all the agents in the simulator.
        """
        if isinstance(agent, PositionObservingAgent) and \
           isinstance(agent, PositionAgent):
            obs = {}
            for other in self.agents.values():
                if other.id == agent.id: continue # Don't observe your own position
                if not isinstance(other, PositionAgent): continue # Can't observe relative position from agents who do not have a position.
                obs[other.id] = other.position - agent.position
            return {'position': obs}
        else:
            return {}
    
    @property
    def null_value(self):
        return np.array([-self.position_state.region, -self.position_state.region])

class PositionRestrictedRelativePositionObserver(RelativePositionObserver):
    """
    Observe the relative position of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value.
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}


class GridPositionBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The values of the cells are as such:
        Out of bounds  : -1
        Empty          :  0
        Agent occupied : 1
    
    position (PositionState):
        The position state handler, which contains the region.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, agents=None, **kwargs):
        self.position = position
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
        self.agents = agents
        from gym.spaces import Box
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, 1, (agent.agent_view*2+1, agent.agent_view*2+1), np.int)

    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, PositionAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Can only observe position of PositionAgents
                if not (isinstance(other_agent, LifeAgent) and other_agent.is_alive): continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff] = 1 # There is an agent at this location.

            return {'position': signal}
        else:
            return {}

class CorridorPositionBasedObserver:
    """
    Agents observe an array of size agent_view centered on their position. The
    values of the cells are as such:
        Empty: 0
        Occupied: 1
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        from gym.spaces import MultiBinary
        for agent in agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = MultiBinary(agent.agent_view*2+1)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Observe 0 if the grid square is empty and 1 if it is occupied.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, PositionAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros(my_agent.agent_view*2+1)
            for other in self.agents.values():
                if other.id == my_agent.id: continue
                if not isinstance(other, PositionAgent) or other.position is None: continue
                diff = other.position - my_agent.position
                if -my_agent.agent_view <= diff <= my_agent.agent_view:
                    signal[diff + my_agnet.view] = 1
        else:
            return {}

class GridPositionTeamBasedObserver:
    """
    Agents observe a grid of size agent_view centered on their
    position. The observation contains one channel per team, where the value of
    the cell is the number of agents on that team that occupy that square. -1
    indicates out of bounds.
    
    position (PositionState):
        The position state handler, which contains the region.

    team_state (TeamState):
        The team state handler, which contains the number of teams.
    
    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, position=None, team_state=None, agents=None, **kwargs):
        self.position = position
        self.team_state = team_state
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
            assert isinstance(agent, TeamAgent)
        self.agents = agents

        from gym.spaces import Box
        for agent in self.agents.values():
            if isinstance(agent, AgentObservingAgent) and \
               isinstance(agent, PositionAgent) and \
               isinstance(agent, PositionObservingAgent):
                agent.observation_space['position'] = Box(-1, np.inf, (agent.agent_view*2+1, agent.agent_view*2+1, self.team_state.number_of_teams), np.int)
    
    def get_obs(self, my_agent, **kwargs):
        """
        Generate an observation of other agents in the grid surrounding this agent's
        position. Each team has its own channel and the value represents the number
        of agents of that team occupying the same square.
        """
        if isinstance(my_agent, AgentObservingAgent) and \
           isinstance(my_agent, TeamAgent) and \
           isinstance(my_agent, PositionAgent) and \
           isinstance(my_agent, PositionObservingAgent):
            signal = np.zeros((my_agent.agent_view*2+1, my_agent.agent_view*2+1))

            # --- Determine the boundaries of the agents' grids --- #
            # For left and top, we just do: view - x,y >= 0
            # For the right and bottom, we just do region - x,y - 1 - view > 0
            if my_agent.agent_view - my_agent.position[0] >= 0: # Top end
                signal[0:my_agent.agent_view - my_agent.position[0], :] = -1
            if my_agent.agent_view - my_agent.position[1] >= 0: # Left end
                signal[:, 0:my_agent.agent_view - my_agent.position[1]] = -1
            if self.position.region - my_agent.position[0] - my_agent.agent_view - 1 < 0: # Bottom end
                signal[self.position.region - my_agent.position[0] - my_agent.agent_view - 1:,:] = -1
            if self.position.region - my_agent.position[1] - my_agent.agent_view - 1 < 0: # Right end
                signal[:, self.position.region - my_agent.position[1] - my_agent.agent_view - 1:] = -1

            # Repeat the boundaries signal for all teams
            signal = np.repeat(signal[:, :, np.newaxis], self.team_state.number_of_teams, axis=2)

            # --- Determine the positions of all the other alive agents --- #
            for other_id, other_agent in self.agents.items():
                if other_id == my_agent.id: continue # Don't observe yourself
                if not isinstance(other_agent, PositionAgent): continue # Cannot observe agent without position
                if not isinstance(other_agent, TeamAgent): continue # Cannot observe agent without team.
                if not (isinstance(other_agent, LifeAgent) and other_agent.is_alive): continue # Can only observe alive agents
                r_diff = other_agent.position[0] - my_agent.position[0]
                c_diff = other_agent.position[1] - my_agent.position[1]
                if -my_agent.agent_view <= r_diff <= my_agent.agent_view and -my_agent.agent_view <= c_diff <= my_agent.agent_view:
                    r_diff += my_agent.agent_view
                    c_diff += my_agent.agent_view
                    signal[r_diff, c_diff, other_agent.team] += 1

            return {'position': signal}
        else:
            return {}

class SpeedObserver:
    """
    Observe the speed of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, SpeedAngleObservingAgent):
                agent.observation_space['speed'] = Dict({
                    other.id: Box(-1, other.max_speed, (1,)) for other in self.agents.values() if isinstance(other, SpeedAngleAgent)
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the speed of all the agents in the simulator.
        """
        if isinstance(agent, SpeedAngleObservingAgent):
            return {'speed': {other.id: other.speed for other in self.agents.values() if isinstance(other, SpeedAngleAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return -1

class PositionRestrictedSpeedObserver(SpeedObserver):
    """
    Observe the speed of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}


class AngleObserver:
    """
    Observe the angle of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, SpeedAngleObservingAgent):
                agent.observation_space['ground_angle'] = Dict({
                    other.id: Box(-1, 360, (1,)) for other in agents.values() if isinstance(other, SpeedAngleAgent)
                })

    def get_obs(self, agent, **kwargs):
        """
        Get the angle of all the agents in the simulator.
        """
        if isinstance(agent, SpeedAngleObservingAgent):
            return {'ground_angle': {other.id: other.ground_angle for other in self.agents.values() if isinstance(other, SpeedAngleAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return -1

class PositionRestrictedAngleObserver(AngleObserver):
    """
    Observe the angle of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}


class VelocityObserver:
    """
    Observe the velocity of all the agents in the simulator.
    """
    def __init__(self, agents=None, **kwargs):
        self.agents = agents
        from gym.spaces import Dict, Box
        for agent in agents.values():
            if isinstance(agent, VelocityObservingAgent):
                agent.observation_space['velocity'] = Dict({
                    other.id: Box(-agent.max_speed, agent.max_speed, (2,)) for other in agents.values() if isinstance(other, VelocityAgent)
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the velocity of all the agents in the simulator.
        """
        if isinstance(agent, VelocityObservingAgent):
            return {'velocity': {agent.id: agent.velocity for agent in self.agents.values() if isinstance(agent, VelocityAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return np.zeros(2)

class PositionRestrictedVelocityObserver(VelocityObserver):
    """
    Observe the velocity of each agent in the simulator if that agent is close
    enough to the observing agent. If it is too far, then observe a null value.
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}



# -------------------------------- #
# --- Resources and Harvesting --- #
# -------------------------------- #

class GridResourceObserver:
    """
    Agents observe a grid of size resource_view centered on their
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
                agent.observation_space['resources'] = Box(0, self.resources.max_value, (agent.resource_view*2+1, agent.resource_view*2+1), np.float)

    def get_obs(self, agent, **kwargs):
        """
        These cells are filled with the values of the resources surrounding the
        agent's position.
        """
        if isinstance(agent, ResourceObservingAgent):
            signal = -np.ones((agent.resource_view*2+1, agent.resource_view*2+1))

            # Derived by considering each square in the resources as an "agent" and
            # then applied the agent diff logic from above. The resulting for-loop
            # can be written in the below vectorized form.
            (r,c) = agent.position
            r_lower = max([0, r-agent.resource_view])
            r_upper = min([self.resources.region-1, r+agent.resource_view])+1
            c_lower = max([0, c-agent.resource_view])
            c_upper = min([self.resources.region-1, c+agent.resource_view])+1
            signal[(r_lower+agent.resource_view-r):(r_upper+agent.resource_view-r),(c_lower+agent.resource_view-c):(c_upper+agent.resource_view-c)] = \
                self.resources.resources[r_lower:r_upper, c_lower:c_upper]
            return {'resources': signal}
        else:
            return {}



# ------------ #
# --- Team --- #
# ------------ #

class TeamObserver:
    """
    Observe the team of each agent in the simulator.
    """
    def __init__(self, team=None, agents=None, **kwargs):
        self.team = team
        self.agents = agents
    
        from gym.spaces import Box, Dict
        for agent in agents.values():
            if isinstance(agent, TeamObservingAgent):
                agent.observation_space['team'] = Dict({
                    other.id: Box(-1, self.team.number_of_teams, (1,), np.int) for other in agents.values() if isinstance(other, TeamAgent)
                })
    
    def get_obs(self, agent, **kwargs):
        """
        Get the team of each agent in the simulator.
        """
        if isinstance(agent, TeamObservingAgent):
            return {'team': {other.id: self.agents[other.id].team for other in self.agents.values() if isinstance(other, TeamAgent)}}
        else:
            return {}
    
    @property
    def null_value(self):
        return -1

class PositionRestrictedTeamObserver(TeamObserver):
    """
    Observe the team of each agent in the simulator if that agent is close enough
    to the observing agent. If it is too far, then observe a null value
    """
    def get_obs(self, agent, **kwargs):
        obs = super().get_obs(agent)
        if obs:
            obs_key = next(iter(obs))
            return {obs_key: obs_filter(obs[obs_key], agent, self.agents, self.null_value)}
        else:
            return {}
