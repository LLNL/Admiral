
from gym.spaces import Box, MultiBinary, Discrete, Dict
import numpy as np
import pytest

from admiral.envs import Agent
from admiral.envs.corridor import MultiCorridor as Corridor
from admiral.managers import TurnBasedManager

def test_init():
    env = Corridor.build()
    wrapped_env = TurnBasedManager(env)
    assert wrapped_env.env == env
    assert wrapped_env.unwrapped == env
    assert wrapped_env.agents == env.agents
    assert next(wrapped_env.agent_order) == 'agent0'
    assert next(wrapped_env.agent_order) == 'agent1'
    assert next(wrapped_env.agent_order) == 'agent2'
    assert next(wrapped_env.agent_order) == 'agent3'
    assert next(wrapped_env.agent_order) == 'agent4'

def test_reset_and_step():
    np.random.seed(24)
    env = TurnBasedManager(Corridor.build())

    obs = env.reset()
    assert env.unwrapped.corridor[4].id == 'agent3'
    assert env.unwrapped.corridor[5].id == 'agent4'
    assert env.unwrapped.corridor[6].id == 'agent2'
    assert env.unwrapped.corridor[7].id == 'agent1'
    assert env.unwrapped.corridor[8].id == 'agent0'
    assert env.done_agents == set()
    assert obs == {'agent0': {'left': [True], 'position': 8, 'right': [False]}}
    
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent1': {'left': [True], 'position': 7, 'right': [False]}}
    assert reward == {'agent1': 0}
    assert done == {'agent1': False, '__all__': False}
    
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent2': {'left': [True], 'position': 6, 'right': [False]}}
    assert reward == {'agent2': 0}
    assert done == {'agent2': False, '__all__': False}
    
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 4, 'right': [True]}}
    assert reward == {'agent3': 0}
    assert done == {'agent3': False, '__all__': False}
    
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent4': {'left': [True], 'position': 5, 'right': [False]}}
    assert reward == {'agent4': -2}
    assert done == {'agent4': False, '__all__': False}
    
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {
        'agent0': {'left': [True], 'position': 9, 'right': [False]},
        'agent1': {'left': [True], 'position': 8, 'right': [False]}}
    assert reward == {'agent0': 100, 'agent1': -1}
    assert done == {'agent0': True, 'agent1': False, '__all__': False}
    
    with pytest.raises(AssertionError):
        env.step({'agent0': Corridor.Actions.STAY})
        
    obs, reward, done, info = env.step({'agent1': Corridor.Actions.STAY})
    assert obs == {'agent2': {'left': [True], 'position': 7, 'right': [True]}}
    assert reward == {'agent2': -1,}
    assert done == {'agent2': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.LEFT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 4, 'right': [False]}}
    assert reward == {'agent3': -5}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.STAY for agent_id in obs})
    assert obs == {'agent4': {'left': [False], 'position': 6, 'right': [True]}}
    assert reward == {'agent4': -3}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.LEFT for agent_id in obs})
    assert obs == {'agent1': {'left': [True], 'position': 8, 'right': [False]}}
    assert reward == {'agent1': -1}
    assert done == {'agent1': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent2': {'left': [False], 'position': 7, 'right': [False]}}
    assert reward == {'agent2': -5}
    assert done == {'agent2': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 4, 'right': [True]}}
    assert reward == {'agent3': -1}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent4': {'left': [True], 'position': 5, 'right': [False]}}
    assert reward == {'agent4': -3}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.LEFT for agent_id in obs})
    assert obs == {
        'agent1': {'left': [True], 'position': 9, 'right': [False]},
        'agent2': {'left': [False], 'position': 8, 'right': [False]}}
    assert reward == {'agent1': 100, 'agent2': -1}
    assert done == {'agent1': True, 'agent2': False, '__all__': False}
    
    with pytest.raises(AssertionError):
        env.step({'agent1': Corridor.Actions.STAY})
        
    obs, reward, done, info = env.step({'agent2': Corridor.Actions.STAY})
    assert obs == {'agent3': {'left': [False], 'position': 4, 'right': [True]}}
    assert reward == {'agent3': -7,}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.LEFT for agent_id in obs})
    assert obs == {'agent4': {'left': [False], 'position': 5, 'right': [False]}}
    assert reward == {'agent4': -5,}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent2': {'left': [False], 'position': 8, 'right': [False]}}
    assert reward == {'agent2': -1,}
    assert done == {'agent2': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 3, 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent4': {'left': [False], 'position': 6, 'right': [False]}}
    assert reward == {'agent4': -1,}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {
        'agent2': {'left': [False], 'position': 9, 'right': [False]},
        'agent3': {'left': [False], 'position': 4, 'right': [False]}}
    assert reward == {'agent2': 100, 'agent3': -1}
    assert done == {'agent2': True, 'agent3': False, '__all__': False}
    
    with pytest.raises(AssertionError):
        env.step({'agent2': Corridor.Actions.STAY})
        
    obs, reward, done, info = env.step({'agent3': Corridor.Actions.RIGHT})
    assert obs == {'agent4': {'left': [False], 'position': 7, 'right': [False]}}
    assert reward == {'agent4': -1,}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 5, 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent4': {'left': [False], 'position': 8, 'right': [False]}}
    assert reward == {'agent4': -1,}
    assert done == {'agent4': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 6, 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {
        'agent4': {'left': [False], 'position': 9, 'right': [False]},
        'agent3': {'left': [False], 'position': 7, 'right': [False]}}
    assert reward == {'agent4': 100, 'agent3': -1}
    assert done == {'agent4': True, 'agent3': False, '__all__': False}
    
    with pytest.raises(AssertionError):
        env.step({'agent4': Corridor.Actions.STAY})
        
    obs, reward, done, info = env.step({'agent3': Corridor.Actions.RIGHT})
    assert obs == {'agent3': {'left': [False], 'position': 8, 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}
        
    obs, reward, done, info = env.step({agent_id: Corridor.Actions.RIGHT for agent_id in obs})
    assert obs == {'agent3': {'left': [False], 'position': 9, 'right': [False]}}
    assert reward == {'agent3': 100,}
    assert done == {'agent3': True, '__all__': True}
