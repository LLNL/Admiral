
from matplotlib import pyplot as plt
import numpy as np

from admiral.component_envs.team import TeamAgent
from admiral.component_envs.observer import ObservingAgent
from admiral.component_envs.position import GridPositionTeamsComponent, GridPositionAgent
from admiral.component_envs.movement import GridMovementComponent, GridMovementAgent
from admiral.component_envs.rewarder import RewarderComponent
from admiral.component_envs.done_conditioner import DoneConditioner
from admiral.envs import AgentBasedSimulation

# TODO: This is much better suited as a unit test.

class ObservingTeamMovementAgent(ObservingAgent, TeamAgent, GridPositionAgent, GridMovementAgent):
    pass

class SimpleGridObservations(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        self.position = GridPositionTeamsComponent(**kwargs)
        self.movement = GridMovementComponent(**kwargs)
        self.rewarder = RewarderComponent(**kwargs)
        self.done_conditioner = DoneConditioner(**kwargs)

        self.finalize()

    def reset(self, **kwargs):
        self.position.reset(**kwargs)

        return {'agent0': self.position.get_obs('agent0')}
    
    def step(self, action_dict, **kwargs):
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            if 'move' in action:
                self.movement.act(agent, action['move'])

        return {'agent0': self.position.get_obs('agent0')}
    
    def render(self, fig=None, **kwargs):
        fig.clear()
        shape = {agent.id: team_shapes[agent.team] for agent in self.agents.values()}
        self.position.render(fig=fig, shape_dict=shape, **kwargs)
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        return self.position.get_obs(agent_id, **kwargs)
    
    def get_reward(self, agent_id, **kwargs):
        return self.rewarder.get_reward(agent_id, **kwargs)
    
    def get_done(self, agent_id, **kwargs):
        return self.done_conditioner.get_done(agent_id, **kwargs)
    
    def get_all_done(self, agent_id, **kwargs):
        return self.done_conditioner.get_all_done(**kwargs)
    
    def get_info(self, agent_id, **kwargs):
        return {}

team_shapes = {
    1: 'o',
    2: 's',
    3: 'd'
}

agents = {
    'agent0': ObservingTeamMovementAgent(id='agent0', team=1, view=1, move=1, starting_position=np.array([2, 1])),
    'agent1': ObservingTeamMovementAgent(id='agent1', team=1, view=1, move=0, starting_position=np.array([2, 2])),
    'agent2': ObservingTeamMovementAgent(id='agent2', team=2, view=1, move=0, starting_position=np.array([0, 4])),
    'agent3': ObservingTeamMovementAgent(id='agent3', team=2, view=1, move=0, starting_position=np.array([0, 0])),
    'agent4': ObservingTeamMovementAgent(id='agent4', team=3, view=1, move=0, starting_position=np.array([4, 0])),
    'agent5': ObservingTeamMovementAgent(id='agent5', team=3, view=1, move=0, starting_position=np.array([4, 4])),
}
env = SimpleGridObservations(
    region=5,
    agents=agents,
    number_of_teams=3
)
obs = env.reset()
fig = plt.gcf()
env.render(fig=fig)
print(obs['agent0'])
print(obs['agent0'])
print(obs['agent0'])
print()

obs = env.step({'agent0': {'move': np.array([-1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, 1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, 1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, -1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([0, -1])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

obs = env.step({'agent0': {'move': np.array([-1, 0])}})
env.render(fig=fig)
print(obs['agent0'][:,:,0])
print(obs['agent0'][:,:,1])
print(obs['agent0'][:,:,2])
print()

plt.show()