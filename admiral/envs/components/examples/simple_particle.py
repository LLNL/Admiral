
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.state import VelocityState, ContinuousPositionState
from admiral.envs.components.actor import AccelerationMovementActor, ContinuousCollisionActor
from admiral.envs.components.observer import VelocityObserver, PositionObserver
from admiral.envs.components.agent import VelocityAgent, PositionAgent, AcceleratingAgent, VelocityObservingAgent, PositionObservingAgent, ActingAgent, CollisionAgent
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class ParticleAgent(VelocityAgent, PositionAgent, AcceleratingAgent, VelocityObservingAgent, PositionObservingAgent, CollisionAgent): pass
class FixedLandmark(PositionAgent): pass
class MovingLandmark(VelocityAgent, PositionAgent): pass

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
        
        # Observer
        self.velocity_observer = VelocityObserver(**kwargs)
        self.position_observer = PositionObserver(position=self.position_state, **kwargs)
    
        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.velocity_state.reset(**kwargs)
    
    def step(self, action_dict, **kwargs):
        for agent, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent], action.get("accelerate", np.zeros(2)), **kwargs)
            self.velocity_state.apply_friction(self.agents[agent], **kwargs)
            
        self.collision_actor.detect_collisions_and_modify_states(**kwargs)
        
        if 'moving_landmark0' in self.agents:
            self.move_actor.process_move(self.agents['moving_landmark0'], np.ones(2), **kwargs)

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
        agents_size = [agent.render_size for agent in self.agents.values() if isinstance(agent, ParticleAgent)]
        mscatter(agents_x, agents_y, ax=ax, m='o', s=agents_size, edgecolor='black', facecolor='gray')

        landmark_x = [agent.position[0] for agent in self.agents.values() if isinstance(agent, (FixedLandmark, MovingLandmark))]
        landmark_y = [agent.position[1] for agent in self.agents.values() if isinstance(agent, (FixedLandmark, MovingLandmark))]
        mscatter(landmark_x, landmark_y, ax=ax, m='o', s=100, edgecolor='black', facecolor='black')

        plt.plot()
        plt.pause(1e-6)

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.position_observer.get_obs(agent),
            **self.velocity_observer.get_obs(agent),
        }

    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        pass

    def get_all_done(self, **kwargs):
        pass

    def get_info(self, agent_id, **kwargs):
        pass

if __name__ == "__main__":
    # agents = {f'agent{i}': ParticleAgent(
    #     id=f'agent{i}',
    #     max_speed=1,
    #     max_acceleration=0.25,
    #     initial_velocity=np.ones(2),
    #     mass=1,
    #     size=1,
    # ) for i in range(10)}

    # env = ParticleEnv(
    #     agents=agents,
    #     region=20,
    #     friction=0.1
    # )
    # fig = plt.figure()
    # env.reset()
    # env.render(fig=fig)
    # print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})

    # for _ in range(24):
    #     env.step({agent.id: {} for agent in agents.values()})
    #     env.render(fig=fig)
    
    agents = {f'agent{i}': ParticleAgent(
        id=f'agent{i}',
        max_speed=.1,
        max_acceleration=0.1,
        mass=1,
        size=1,
    ) for i in range(20)}
    agents = {**agents, 
        f'fixed_landmark0': FixedLandmark(id='fixed_landmark0'),
        f'moving_landmark0': MovingLandmark(id='moving_landmark0', max_speed=1),
    }

    env = ParticleEnv(
        agents=agents,
        region=20,
        friction=0.0
    )
    fig = plt.figure()
    env.reset()
    env.render(fig=fig)

    for _ in range(200):
        env.step({agent.id: agent.action_space.sample() for agent in agents.values() if isinstance(agent, ActingAgent)})
        env.render(fig=fig)
