
from gym.spaces import Box, MultiBinary, Discrete, Dict
import numpy as np
import pytest

from admiral.envs import Agent
from admiral.envs.corridor import MultiCorridor as Corridor
from admiral.managers import AllStepManager

def test_init():
    env = Corridor.build()
    wrapped_env = AllStepManager(env)
    assert wrapped_env.env == env
    assert wrapped_env.unwrapped == env
    assert wrapped_env.agents == env.agents

def test_reset_and_step():
    np.random.seed(24)
    env = AllStepManager(Corridor.build())

    obs = env.reset()
    assert env.unwrapped.corridor[4].id == 'agent3'
    assert env.unwrapped.corridor[5].id == 'agent4'
    assert env.unwrapped.corridor[6].id == 'agent2'
    assert env.unwrapped.corridor[7].id == 'agent1'
    assert env.unwrapped.corridor[8].id == 'agent0'
    assert env.done_agents == set()
    assert obs['agent0'] == {'left': [True], 'position': 8, 'right': [False]}
    assert obs['agent1'] == {'left': [True], 'position': 7, 'right': [True]}
    assert obs['agent2'] == {'left': [True], 'position': 6, 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': 4, 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': 5, 'right': [True]}
    

    obs, reward, done, _ = env.step({
        'agent0': Corridor.Actions.RIGHT,
        'agent1': Corridor.Actions.RIGHT,
        'agent2': Corridor.Actions.RIGHT,
        'agent3': Corridor.Actions.RIGHT,
        'agent4': Corridor.Actions.RIGHT,
    })

    assert obs['agent0'] == {'left': [True], 'position': 9, 'right': [False]}
    assert obs['agent1'] == {'left': [True], 'position': 8, 'right': [False]}
    assert obs['agent2'] == {'left': [True], 'position': 7, 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': 4, 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': 6, 'right': [True]}
    assert reward['agent0'] == 100
    assert reward['agent1'] == -1
    assert reward['agent2'] == -1
    assert reward['agent3'] == -5
    assert reward['agent4'] == -3
    assert done['agent0'] == True
    assert done['agent1'] == False
    assert done['agent2'] == False
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False

    
    with pytest.raises(AssertionError):
        env.step({
            'agent0': Corridor.Actions.RIGHT,
            'agent1': Corridor.Actions.STAY,
            'agent2': Corridor.Actions.LEFT,
            'agent3': Corridor.Actions.STAY,
            'agent4': Corridor.Actions.LEFT,
        })
    
    obs, reward, done, _ = env.step({
        'agent1': Corridor.Actions.STAY,
        'agent2': Corridor.Actions.LEFT,
        'agent3': Corridor.Actions.STAY,
        'agent4': Corridor.Actions.LEFT,
    })

    assert 'agent0' not in obs
    assert obs['agent1'] == {'left': [True], 'position': 8, 'right': [False]}
    assert obs['agent2'] == {'left': [False], 'position': 7, 'right': [True]}
    assert obs['agent3'] == {'left': [False], 'position': 4, 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': 5, 'right': [False]}
    assert 'agent0' not in reward
    assert reward['agent1'] == -1
    assert reward['agent2'] == -5
    assert reward['agent3'] == -1
    assert reward['agent4'] == -3
    assert 'agent0' not in done
    assert done['agent1'] == False
    assert done['agent2'] == False
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False

    
    obs, reward, done, _ = env.step({
        'agent1': Corridor.Actions.RIGHT,
        'agent2': Corridor.Actions.RIGHT,
        'agent3': Corridor.Actions.RIGHT,
        'agent4': Corridor.Actions.LEFT,
    })

    assert obs['agent1'] == {'left': [True], 'position': 9, 'right': [False]}
    assert obs['agent2'] == {'left': [False], 'position': 8, 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': 4, 'right': [True]}
    assert obs['agent4'] == {'left': [True], 'position': 5, 'right': [False]}
    assert reward['agent1'] == 100
    assert reward['agent2'] == -1
    assert reward['agent3'] == -7
    assert reward['agent4'] == -7
    assert done['agent1'] == True
    assert done['agent2'] == False
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False

    
    with pytest.raises(AssertionError):
        env.step({
            'agent1': Corridor.Actions.STAY,
            'agent2': Corridor.Actions.STAY,
            'agent3': Corridor.Actions.LEFT,
            'agent4': Corridor.Actions.RIGHT,
        })
    
    obs, reward, done, _ = env.step({
        'agent2': Corridor.Actions.STAY,
        'agent3': Corridor.Actions.LEFT,
        'agent4': Corridor.Actions.RIGHT,
    })

    assert 'agent1' not in obs
    assert obs['agent2'] == {'left': [False], 'position': 8, 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': 3, 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': 6, 'right': [False]}
    assert 'agent1' not in reward
    assert reward['agent2'] == -1
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert 'agent1' not in done
    assert done['agent2'] == False
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False

    
    obs, reward, done, _ = env.step({
        'agent2': Corridor.Actions.RIGHT,
        'agent3': Corridor.Actions.RIGHT,
        'agent4': Corridor.Actions.RIGHT,
    })

    assert obs['agent2'] == {'left': [False], 'position': 9, 'right': [False]}
    assert obs['agent3'] == {'left': [False], 'position': 4, 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': 7, 'right': [False]}
    assert reward['agent2'] == 100
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert done['agent2'] == True
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False
        
    
    with pytest.raises(AssertionError):
        env.step({
            'agent2': Corridor.Actions.STAY,
            'agent3': Corridor.Actions.RIGHT,
            'agent4': Corridor.Actions.RIGHT,
        })

    obs, reward, done, _ = env.step({
        'agent3': Corridor.Actions.RIGHT,
        'agent4': Corridor.Actions.RIGHT,
    })

    assert 'agent2' not in obs
    assert obs['agent3'] == {'left': [False], 'position': 5, 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': 8, 'right': [False]}
    assert 'agent2' not in reward
    assert reward['agent3'] == -1
    assert reward['agent4'] == -1
    assert 'agent2' not in done
    assert done['agent3'] == False
    assert done['agent4'] == False
    assert done['__all__'] == False


    obs, reward, done, _ = env.step({
        'agent3': Corridor.Actions.RIGHT,
        'agent4': Corridor.Actions.RIGHT,
    })

    assert obs['agent3'] == {'left': [False], 'position': 6, 'right': [False]}
    assert obs['agent4'] == {'left': [False], 'position': 9, 'right': [False]}
    assert reward['agent3'] == -1
    assert reward['agent4'] == 100
    assert done['agent3'] == False
    assert done['agent4'] == True
    assert done['__all__'] == False

    
    with pytest.raises(AssertionError):
        env.step({
            'agent3': Corridor.Actions.RIGHT,
            'agent4': Corridor.Actions.STAY,
        })

    obs, reward, done, _ = env.step({
        'agent3': Corridor.Actions.RIGHT,
    })
    
    assert 'agent4' not in obs
    assert obs == {'agent3': {'left': [False], 'position': 7, 'right': [False]}}
    assert 'agent4' not in reward
    assert reward == {'agent3': -1,}
    assert 'agent4' not in done
    assert done == {'agent3': False, '__all__': False}


    obs, reward, done, _ = env.step({
        'agent3': Corridor.Actions.RIGHT,
    })
    
    assert obs == {'agent3': {'left': [False], 'position': 8, 'right': [False]}}
    assert reward == {'agent3': -1,}
    assert done == {'agent3': False, '__all__': False}


    obs, reward, done, _ = env.step({
        'agent3': Corridor.Actions.RIGHT,
    })
    
    assert obs == {'agent3': {'left': [False], 'position': 9, 'right': [False]}}
    assert reward == {'agent3': 100}
    assert done == {'agent3': True, '__all__': True}
