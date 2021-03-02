
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from admiral.envs.components.state import TeamState, GridPositionState, LifeState, GridResourceState
from admiral.envs.components.observer import GridPositionTeamBasedObserver, GridResourceObserver
from admiral.envs.components.actor import GridMovementActor, PositionTeamBasedAttackActor, GridResourcesActor
from admiral.envs.components.done import TeamDeadDone
from admiral.envs.components.agent import TeamAgent, PositionAgent, LifeAgent, AgentObservingAgent, PositionObservingAgent, ResourceObservingAgent, GridMovementAgent, AttackingAgent, HarvestingAgent
from admiral.envs import AgentBasedSimulation
from admiral.tools.matplotlib_utils import mscatter

class PreyAgent(PositionAgent, TeamAgent, GridMovementAgent, LifeAgent, AgentObservingAgent, HarvestingAgent, ResourceObservingAgent, PositionObservingAgent):
    pass

class PredatorAgent(PositionAgent, TeamAgent, GridMovementAgent, LifeAgent, AgentObservingAgent, AttackingAgent, PositionObservingAgent):
    pass

class PredatorPreyEnvGridBased(AgentBasedSimulation):
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']

        # State components
        self.position_state = GridPositionState(**kwargs)
        self.life_state = LifeState(**kwargs)
        self.resource_state = GridResourceState(**kwargs)
        self.team_state = TeamState(**kwargs)

        # Observer components
        self.position_observer = GridPositionTeamBasedObserver(position=self.position_state, team_state=self.team_state, **kwargs)
        self.resource_observer = GridResourceObserver(resources=self.resource_state, **kwargs)

        # Actor components
        self.move_actor = GridMovementActor(position_state=self.position_state, **kwargs)
        self.resource_actor = GridResourcesActor(resources=self.resource_state, **kwargs)
        self.attack_actor = PositionTeamBasedAttackActor(**kwargs)

        # Done components
        self.done = TeamDeadDone(**kwargs)

        self.finalize()
    
    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.resource_state.reset(**kwargs)
        self.life_state.reset(**kwargs)

    def step(self, action_dict, **kwargs):
        # Process harvesting
        for agent_id, action in action_dict.items():
            agent = self.agents[agent_id]
            harvested_amount = self.resource_actor.process_harvest(agent, action.get('harvest', 0), **kwargs)
            if harvested_amount is not None:
                self.life_state.modify_health(agent, harvested_amount)
        
        # Process attacking
        for agent_id, action in action_dict.items():
            attacking_agent = self.agents[agent_id]
            attacked_agent = self.attack_actor.process_attack(attacking_agent, action.get('attack', False), **kwargs)
            if attacked_agent is not None:
                self.life_state.modify_health(attacked_agent, -attacking_agent.attack_strength)
                self.life_state.modify_health(attacking_agent, attacking_agent.attack_strength)

        # Process movement
        for agent_id, action in action_dict.items():
            self.move_actor.process_move(self.agents[agent_id], action.get('move', np.zeros(2)), **kwargs)

        # Apply entropy to all agents
        for agent_id in action_dict:
            self.life_state.apply_entropy(self.agents[agent_id])

        # Regrow the resources
        self.resource_state.regrow()
    
    def render(self, fig=None, **kwargs):
        fig.clear()

        # Draw the resources
        ax = fig.gca()
        ax = sns.heatmap(np.flipud(self.resource_state.resources), ax=ax, cmap='Greens')

        # Draw the agents
        render_condition = {agent.id: agent.is_alive for agent in self.agents.values()}
        shape_dict = {agent.id: 'o' if agent.team == 1 else 's' for agent in self.agents.values()}
        ax.set(xlim=(0, self.position_state.region), ylim=(0, self.position_state.region))
        ax.set_xticks(np.arange(0, self.position_state.region, 1))
        ax.set_yticks(np.arange(0, self.position_state.region, 1))
        ax.grid()

        agents_x = [agent.position[1] + 0.5 for agent in self.agents.values() if render_condition[agent.id]]
        agents_y = [self.position_state.region - 0.5 - agent.position[0] for agent in self.agents.values() if render_condition[agent.id]]
        shape = [shape_dict[agent_id] for agent_id in shape_dict if render_condition[agent_id]]
        mscatter(agents_x, agents_y, ax=ax, m=shape, s=200, edgecolor='black', facecolor='gray')

        plt.plot()
        plt.pause(1e-6)
    
    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return {
            **self.position_observer.get_obs(agent),
            **self.resource_observer.get_obs(agent),
        }
    
    def get_reward(self, agent_id, **kwargs):
        pass

    def get_done(self, agent_id, **kwargs):
        return self.done.get_done(self.agents[agent_id])
    
    def get_all_done(self, **kwargs):
        return self.done.get_all_done(**kwargs)
    
    def get_info(self, *args, **kwargs):
        return {}

if __name__ == '__main__':
    prey =      {f'prey{i}':     PreyAgent(    id=f'prey{i}',     agent_view=5, team=0, move_range=1, max_harvest=0.5, resource_view=5) for i in range(7)}
    predators = {f'predator{i}': PredatorAgent(id=f'predator{i}', agent_view=2, team=1, move_range=1, attack_range=1, attack_strength=0.24) for i in range(2)}
    agents = {**prey, **predators}
    region = 10
    env = PredatorPreyEnvGridBased(
        region=region,
        agents=agents,
        number_of_teams=2,
        entropy=0.05
    )
    env.reset()
    print({agent_id: env.get_obs(agent_id) for agent_id in env.agents})
    fig = plt.gcf()
    env.render(fig=fig)

    for _ in range(50):
        action_dict = {agent.id: agent.action_space.sample() for agent in env.agents.values() if agent.is_alive}
        env.step(action_dict)
        env.render(fig=fig)
        print(env.get_all_done())
