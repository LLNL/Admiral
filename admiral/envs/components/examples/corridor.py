
from matplotlib import pyplot as plt
import numpy as np

from admiral.envs.components.state import GridUniquePositionState
from admiral.envs.components.actor import GridMovementActor
from admiral.envs.components.observer import CorridorPositionBasedObserver, PositionObserver
from admiral.envs.components.done import TargetPositionDone
from admiral.envs.components.agent import PositionAgent, GridMovementAgent, PositionObservingAgent, AgentObservingAgent
from admiral.envs import AgentBasedSimulation

class CorridorAgent(PositionAgent, GridMovementAgent, PositionObservingAgent, AgentObservingAgent): pass

class MultiCorridorEnv(AgentBasedSimulation):
    """
    MultiCorridor Environment where multiple agents spawn along
    a corridor and can choose to move left, right, or stay still. The agents
    must learn to move to the right until they reach the end position. The agent
    cannot move to spaces that area already occupied by other agents. An agent
    is done once it moves to the end position.

    The agent can observe its own position. It can also see if the squares
    near it are occupied.
    """
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State
        self.position_state = GridUniquePositionState(**kwargs)

        # Actor
        self.movement_actor = GridMovementActor(position_state=self.position_state, **kwargs)

        # Observer
        self.nearby_observer = CorridorPositionBasedObserver(**kwargs)

        # Done
        self.target_position_done = TargetPositionDone(**kwargs)

        # Rewarder

        self.finalize()

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.reward = {agent_id: 0 for agent_id in self.agents}
    
    def step(self, action_dict, **kwargs):
        for agent_id in action_dict:
            self.reward[agent_id] = 0

        # Process moving
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            amount_moved = self.movement_actor.process_move(agent, action.get('move', np.zeros(2)), **kwargs)
            
            if action['move'][0] == 0:
                self.reward[agent_id] += -1
            elif action['move'][0] == -1 and amount_moved != -1:
                self.reward[agent_id] += -5
            elif action['move'][0] == 1:
                if amount_moved != 1:
                    self.reward[agent_id] += -5
                else:
                    if np.all(agent.position == self.target_position_done.target):
                        self.reward[agent_id] += self.position_state.region ** 2

    def render(self, fig=None, **kwargs):
        """
        Visualize the state of the environment. If a figure is received, then we
        will draw but not actually plot because we assume the caller will do the
        work (e.g. with an Animation object). If there is no figure received, then
        we will draw and plot the environment.
        """
        draw_now = fig is None
        if draw_now:
            from matplotlib import pyplot as plt
            fig = plt.gcf()

        fig.clear()
        ax = fig.gca()
        ax.set(xlim=(-0.5, self.position_state.region - 0.5), ylim=(-0.5, 0.5))
        ax.set_xticks(np.arange(0, self.position_state.region, 1.))
        ax.scatter(np.array(
            [agent.position for agent in self.agents.values()]),
            np.zeros(len(self.agents)),
            marker='s', s=200, c='g'
        )
    
        # if draw_now:
        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.nearby_observer.get_obs(agent, **kwargs)
        }
    
    def get_reward(self, agent_id, **kwargs):
        return self.reward[agent_id]

    def get_done(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        if self.target_position_done.get_done(agent, **kwargs):
            agent.position = None
            return True
        else:
            return False
    
    def get_all_done(self, **kwargs):
        return self.target_position_done.get_all_done(**kwargs)
    
    def get_info(self, **kwargs):
        pass

if __name__ == '__main__':
    agents = {
        f'agent{i}': CorridorAgent(id=f'agent{i}', move_range=1, agent_view=1)
        for i in range(5)
    }
    env = MultiCorridorEnv(
        agents=agents,
        dimension=1,
        region=10,
        target=np.array([9])
    )

    env.reset()
    print(env.get_obs('agent0'))
    env.render()

    done_agents = set()
    for _ in range(50):
        # print(done_agents)
        actions = {agent.id: agent.action_space.sample() for agent in agents.values() if agent.id not in done_agents}
        env.step(actions)
        env.render()
        done_agents.update((agent_id for agent_id in agents if agent_id not in done_agents and env.get_done(agent_id)))
