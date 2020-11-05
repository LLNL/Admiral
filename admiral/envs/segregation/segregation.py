import random

import networkx as nx
import numpy as np

from amber.envs import GettingEnvironment
from amber.envs import Agent

class RedAgent(Agent):
    pass

class GreenAgent(Agent):
    pass

class Segregation(GettingEnvironment):
    def __init__(self, config: dict):
        self.agents = config['agents']
        self.region = config['region']
        self.neighbors = config['neighbors']

    def reset(self) -> dict:
        """
        Returns agent's ids mapped to their observations.
        """
        # Randomly distribute the agents throughout the region. The grid will hold the Agent object
        # as an attribute of the nodes.
        self.grid = nx.generators.grid_2d_graph(self.region, self.region)
        if self.neighbors == 8:
            self.grid.add_edges_from([
                ((x, y), (x+1, y+1))
                for x in range(self.region-1)
                for y in range(self.region-1)
            ] + [
                ((x+1, y), (x, y+1))
                for x in range(self.region-1)
                for y in range(self.region-1)
            ])

        random_indices = np.unravel_index(
             # Create a list of unique "raveled" indices
            np.random.choice(np.arange(self.region**2), size=(len(self.agents),), replace=False),
            (self.region, self.region) # Unravel the indices into coordinates in the grid
        )
        for i, agent in enumerate(self.agents.values()):
            node = random_indices[0][i], random_indices[1][i]
            self.grid.nodes[node]['agent'] = agent
            agent.position = node
        
        # Maintain a list of empty nodes for easy reassignment
        self.empty_nodes = set(node for node, atts in self.grid.nodes.items() if 'agent' not in atts)
        return self._collect_state()

    def step(self, joint_actions: dict):
        """
        The agent can choose one of two actions: (1) it can choose to stay where it is or (2) it can
        choose to move. In this environment, the agent is moved to a random location on the map that
        has a free square.

        Joint actions maps the agent's id to the action it has decided to take.

        Returns agent's ids mapped to their observations, their rewards, their done conditions,
        and their extra information.
        """
        self.done = True
        for agent_id, action in joint_actions.items():
            if action == 1: # Agent chooses to move
                self.done = False

                move_to_node = random.sample(self.empty_nodes, 1)[0]
                agent_moving = self.agents[agent_id]
                move_from_node = agent_moving.position

                agent_moving.position = move_to_node
                del self.grid.nodes[move_from_node]['agent']
                self.grid.nodes[move_to_node]['agent'] = agent_moving
                self.empty_nodes.remove(move_to_node)
                self.empty_nodes.add(move_from_node)
        
        return self._collect_state(), 0, self.done, {}

    def render(self):
        from matplotlib import pyplot as plt
        ax = plt.gca()
        ax.clear()
        pos = dict((n,n) for n in self.grid.nodes)
        colors = []
        for node, atts in self.grid.nodes.items():
            agent = atts.get('agent', None)
            if agent is None:
                colors.append('w')
            elif isinstance(agent, RedAgent):
                colors.append('r')
            else: # Green
                colors.append('g')
        nx.draw_networkx_nodes(self.grid, pos=pos, node_color=colors, ax=ax)
        plt.draw()
        plt.pause(1e-17)
    
    def get_obs(self, agent_id):
        node = self.grid.nodes[self.agents[agent_id].position]
        counts = {'same': 0, 'diff': 0, 'empty': 0}
        for neighbor_node in self.grid.adj[node]: # Loop over the neighbors
            neighbor_atts = self.grid.nodes[neighbor_node] # Get the node attribute for this neighbor
            neighbor_agent = neighbor_atts.get('agent', None)
            if neighbor_agent is None:
                counts['empty'] += 1
            elif type(neighbor_agent) == type(agent):
                counts['same'] += 1
            else: # types are not equal
                counts['diff'] += 1
        return np.fromiter(counts.values(), np.int)

    def get_done(self, agent_id):
        return self.get_all_done()

    def get_all_done(self):
        return self.done

    def get_reward(self, agent_id):
        0

    def get_info(self, agent_id):
        {}

    def _collect_state(self):
        """
        For every agent in the network, collect a count of how many neighbors are the same color,
        different color, and empty.
        """
        observations = {}
        for node, atts in self.grid.nodes.items():
            agent = atts.get('agent', None) # Get the agent that is at this location
            if agent is not None: # Only process if there is an agent there
                # Count the surrounding cells
                counts = {'same': 0, 'diff': 0, 'empty': 0}
                for neighbor_node in self.grid.adj[node]: # Loop over the neighbors
                    neighbor_atts = self.grid.nodes[neighbor_node] # Get the node attribute for this neighbor
                    neighbor_agent = neighbor_atts.get('agent', None)
                    if neighbor_agent is None:
                        counts['empty'] += 1
                    elif type(neighbor_agent) == type(agent):
                        counts['same'] += 1
                    else: # types are not equal
                        counts['diff'] += 1
                observations[agent.id] = np.fromiter(counts.values(), np.int)
        return observations

    @classmethod
    def build(cls, env_config={}):
        """
        Parameters
        ----------
        region: int
            The size of the grid. Default 10.
        
        neighbors: int
            Use either the 4 neighbors to the NSEW or else use the 8 surrounding neighbors. Default 8.

        agents: dict of Agents
            A dictionary of RedAgents and GreenAgents. Key's are id's mapping to Agent object.
            Default: enough Red and Green Agents to cover about 3/4ths of the region.
        """
        config = {}

        # Region
        config['region'] = env_config.get('region', 10)
        
        # Neighbors
        config['neighbors'] = env_config.get('neighbors', 8)

        # Agents
        from gym.spaces import Box, MultiBinary
        if 'agents' in env_config:
            for agent in env_config['agents'].values():
                agent.observation_space = Box(low=0, high=config['neighbors'], shape=(3,), dtype=np.int),
                agent.action_space = MultiBinary(1)
            config['agents'] = env_config['agents']
        else:
            num_agents_each_team = int(config['region'] ** 2 * 0.375)
            agents = {}
            for i in range(num_agents_each_team):
                agents[f'red{i}'] = RedAgent(
                    id=f'red{i}',
                    observation_space=Box(low=0, high=config['neighbors'], shape=(3,), dtype=np.int),
                    action_space=MultiBinary(1)
                )
                agents[f'green{i}'] = GreenAgent(
                    id=f'green{i}',
                    observation_space=Box(low=0, high=config['neighbors'], shape=(3,), dtype=np.int),
                    action_space=MultiBinary(1)
                )
            config['agents'] = agents

        # Build
        return cls(config)
                    

