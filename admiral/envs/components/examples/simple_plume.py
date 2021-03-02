
from admiral.envs.components.state import ContinuousPositionState, PlumeState
from admiral.envs.components.observer import PlumeSampleObserver
from admiral.envs.components.agent import PositionAgent, PlumeSamplingAgent

class SimplePlumeAgent(PositionAgent, PlumeSamplingAgent): pass

class SimplePlumeEnv:
    def __init__(self, **kwargs):
        self.agents = kwargs['agents']
        
        # State
        self.position_state = ContinuousPositionState(**kwargs)
        self.plume_state = PlumeState(**kwargs)

        # Observer
        self.plume_observer = PlumeSampleObserver(plume_state=self.plume_state, **kwargs)

    def reset(self, **kwargs):
        self.position_state.reset(**kwargs)
        self.plume_state.reset(**kwargs)
    
    def render(self, fig=None, **kwargs):
        pass

    def get_obs(self, agent_id, **kwargs):
        agent = self.agents[agent_id]
        return self.plume_observer(agent, **kwargs)

agents = {
    f'agent{i}': SimplePlumeAgent(id=f'agent{i}') for i in range(5)
}

env = SimplePlumeEnv(
    agents=agents,
    region=10,
)

env.reset()
env.render()
env.get_obs('agent0')