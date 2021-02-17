
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.state import VelocityState, ContinuousPositionState
from admiral.envs.components.actor import AccelerationMovementActor, ContinuousCollisionActor
from admiral.envs.components.agent import VelocityAgent, PositionAgent, AcceleratingAgent, CollisionAgent
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class ParticleAgent(VelocityAgent, PositionAgent, AcceleratingAgent, CollisionAgent): pass

class ParticleEnv(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State
        self.position_state = ContinuousPositionState(**kwargs)
        self.velocity_state = VelocityState(**kwargs)

        # Actor
        self.move_actor = AccelerationMovementActor(position_state=self.position_state, \
            velocity_state=self.velocity_state, **kwargs)
        self.collision_actor = ContinuousCollisionActor(position_state=self.position_state, \
            velocity_state=self.velocity_state, **kwargs)
    
        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.velocity_state.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent], action.get("accelerate", np.zeros(2)), **kwargs)
            self.velocity_state.apply_friction(self.agents[agent], **kwargs)
        
        self.render(fig=kwargs['fig'])
        self.collision_actor.detect_collisions_and_modify_states(env=self, **kwargs)

    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()

        # Draw the agents
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))

        agents_x = [agent.position[0] for agent in self.agents.values() if isinstance(agent, ParticleAgent)]
        agents_y = [agent.position[1] for agent in self.agents.values() if isinstance(agent, ParticleAgent)]
        agents_size = [8*agent.render_size for agent in self.agents.values() if isinstance(agent, ParticleAgent)]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=agents_size, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        pass

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        pass

    def get_all_done(self, **kwargs):
        pass

    def get_info(self, agent_id, **kwargs):
        pass

if __name__ == "__main__":
    agents = {
        # 'agent0': ParticleAgent(id='agent0', max_acceleration=0, max_speed=10, size=1, mass=1, initial_velocity=np.array([1, 1]), initial_position=np.array([1,1])),
        # 'agent1': ParticleAgent(id='agent1', max_acceleration=0, max_speed=10, size=1, mass=1, initial_velocity=np.array([-1, 1]), initial_position=np.array([4, 1]))
        'agent0': ParticleAgent(id='agent0', max_acceleration=0, max_speed=10, size=1, mass=1, initial_velocity=np.array([1, 0.5]), initial_position=np.array([1,1])),
        'agent1': ParticleAgent(id='agent1', max_acceleration=0, max_speed=10, size=1, mass=1, initial_velocity=np.array([0, 1]), initial_position=np.array([4, 1]))
    }
    env = ParticleEnv(
        agents=agents,
        region = 8,
        friction=0.0
    )

    fig = plt.figure()
    env.reset()
    env.render(fig=fig)
    x = []

    env.step({agent.id: {'accelerate': np.zeros(2)} for agent in agents.values()}, fig=fig)
    env.render(fig=fig)
    x = [] 

    env.step({agent.id: {'accelerate': np.zeros(2)} for agent in agents.values()}, fig=fig)
    env.render(fig=fig)
    x = [] 

    env.step({agent.id: {'accelerate': np.zeros(2)} for agent in agents.values()}, fig=fig)
    env.render(fig=fig)
    x = [] 

    env.step({agent.id: {'accelerate': np.zeros(2)} for agent in agents.values()}, fig=fig)
    env.render(fig=fig)
    x = [] 

    env.step({agent.id: {'accelerate': np.zeros(2)} for agent in agents.values()}, fig=fig)
    env.render(fig=fig)
    x = [] 
