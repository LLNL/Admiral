
import numpy as np

from admiral.envs.components.agent import LifeAgent, TeamAgent, PositionAgent

class DeadDone:
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    all the agents are dead.

    agents (dict):
        The dictionary of agents. Because the done condition is determined by
        the agent's life status, all agents must be LifeAgents.
    """
    def __init__(self, agents=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, LifeAgent)
        self.agents = agents
    
    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is dead. Otherwise, return False.
        """
        return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Return True if all agents are dead. Otherwise, return False.
        """
        for agent in self.agents.values():
            if agent.is_alive:
                return False
        return True

class TeamDeadDone:
    """
    Dead agents are indicated as done. Additionally, the simulation is over when
    the only agents remaining are on the same team.

    agents (dict):
        The dictionary of agents. Because the done condition is determined by the
        agent's life status, all agents must be LifeAgents; and because the done
        condition is determined by the agents' teams, all agents must be TeamAgents.

    number_of_teams (int):
        The fixed number of teams in this simulation.
    """
    def __init__(self, agents=None, number_of_teams=None, **kwargs):
        for agent in agents.values():
            assert isinstance(agent, TeamAgent)
            assert isinstance(agent, LifeAgent)
        self.agents = agents
        assert type(number_of_teams) is int, "number_of_teams must be a positive integer."
        self.number_of_teams = number_of_teams
    
    def get_done(self, agent, **kwargs):
        """
        Return True if the agent is dead. Otherwise, return False.
        """
        return not agent.is_alive

    def get_all_done(self, **kwargs):
        """
        Return true if the only agent left alive are all on the same team. Otherwise,
        return false.
        """
        team = np.zeros(self.number_of_teams)
        for agent in self.agents.values():
            if agent.is_alive:
                team[agent.team] += 1
        return sum(team != 0) <= 1

class TargetPositionDone:
    """
    The agent is done if it reaches a specific location.

    target (np.array):
        The target position for all agents to reach.

    agents (dict):
        The dictionary of agents.
    """
    def __init__(self, target=None, agents=None, **kwargs):
        self.target = target
        self.agents = agents
    
    def get_done(self, agent, **kwargs):
        """
        Return true if the agent reaches done position.
        """
        if isinstance(agent, PositionAgent):
            return np.all(agent.position == self.target)
        else:
            return True
    
    def get_all_done(self, **kwargs):
        """
        Return true if all agents are done.
        """
        return all((self.get_done(agent) for agent in self.agents.values()))

class TooCloseDone:
    """
    Agents that are too close to each other or too close to the edge of the region
    are indicated as done. If any agent is done, the entire simulation is done.

    position (PositionState):
        The position state handler.

    agents (dict):
        Dictionay of agent objects.
    
    collision_distance (float):
        The threshold for calculating if a collision has occured.
    
    collision_norm (int):
        The norm to use when calculating the collision. For example, you would
        probably want to use 1 in the Grid space but 2 in a Continuous space.
        Default is 2.
    """
    def __init__(self, position=None, agents=None, collision_distance=None, collision_norm=2, **kwargs):
        assert position is not None
        self.position = position
        for agent in agents.values():
            assert isinstance(agent, PositionAgent)
        self.agents = agents
        assert collision_distance is not None
        self.collision_distance = collision_distance
        self.collision_norm = collision_norm
    
    def get_done(self, agent, **kwargs):
        """
        Return true if the agent is too close to another agent or too close to
        the edge of the region.
        """
        #  Collision with region edge
        if np.any(agent.position[0] < self.collision_distance) \
            or np.any(agent.position[0] > self.position.region - self.collision_distance) \
            or np.any(agent.position[1] < self.collision_distance) \
            or np.any(agent.position[1] > self.position.region - self.collision_distance):
            return True

        # Collision with other birds
        for other in self.agents.values():
            if other.id == agent.id: continue # Cannot collide with yourself
            if np.linalg.norm(other.position - agent.position, self.collision_norm) < self.collision_distance:
                return True
        return False
    
    def get_all_done(self, **kwargs):
        """
        Return true if any agent is too close to another agent or too close to
        the edge of the region.
        """
        for agent in self.agents.values():
            if self.get_done(agent):
                return True
        return False
