
from gym.spaces import Box, Dict, MultiBinary
import numpy as np

from admiral.envs.components.examples.predator_prey_example import PredatorPreyEnvGridBased, PreyAgent, PredatorAgent
from admiral.managers import AllStepManager

def test_all_step_grid_based_predator_prey():
    agents = {
        'prey0': PreyAgent(id='prey0', initial_position=np.array([2, 2]), agent_view=4, initial_health=0.5, team=0, move_range=1, max_harvest=0.5, resource_view_range=4),
        'prey1': PreyAgent(id='prey1', initial_position=np.array([2, 2]), agent_view=4, initial_health=0.5, team=0, move_range=1, max_harvest=0.5, resource_view_range=4),
        'predator0': PredatorAgent(id='predator0', initial_position=np.array([0, 0]), agent_view=2, initial_health=0.5, team=1, move_range=1, attack_range=1, attack_strength=2.0)
    }
    original_resources = np.array([
        [0.43, 0.  , 0.  , 0.37, 0.32],
        [0.85, 0.34, 0.47, 0.24, 0.65],
        [0.86, 0.62, 0.45, 0.98, 0.26],
        [0.  , 0.  , 0.9 , 0.  , 0.  ],
        [0.  , 0.18, 0.97, 0.  , 0.94]
    ])
    env = AllStepManager(PredatorPreyEnvGridBased(
        region=5,
        agents=agents,
        number_of_teams=2,
        entropy=0.1,
        original_resources=original_resources,
    ))

    obs = env.reset()
    np.testing.assert_array_equal(obs['prey0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  1.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  1.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey0']['resources'], np.array([
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.43,  0.  ,  0.  ,  0.37,  0.32, -1.,   -1.,  ],
        [-1.,   -1.,    0.85,  0.34,  0.47,  0.24,  0.65, -1.,   -1.,  ],
        [-1.,   -1.,    0.86,  0.62,  0.45,  0.98,  0.26, -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.  ,  0.9 ,  0.  ,  0.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.18,  0.97,  0.  ,  0.94, -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ]
    ]))

    np.testing.assert_array_equal(obs['prey1']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  1.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  1.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['resources'], np.array([
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.43,  0.  ,  0.  ,  0.37,  0.32, -1.,   -1.,  ],
        [-1.,   -1.,    0.85,  0.34,  0.47,  0.24,  0.65, -1.,   -1.,  ],
        [-1.,   -1.,    0.86,  0.62,  0.45,  0.98,  0.26, -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.  ,  0.9 ,  0.  ,  0.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.18,  0.97,  0.  ,  0.94, -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ]
    ]))

    np.testing.assert_array_equal(obs['predator0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.],
        [-1., -1.,  0.,  0.,  0.],
        [-1., -1.,  0.,  0.,  2.]
    ]))
    np.testing.assert_array_equal(obs['predator0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.],
        [-1., -1.,  0.,  0.,  0.],
        [-1., -1.,  0.,  0.,  0.]
    ]))


    obs, reward, done, _ = env.step({
        'prey0': {'move': np.array([-1, -1]), 'harvest': 0.},
        'prey1': {'move': np.array([-1,  0]), 'harvest': 0.},
        'predator0': {'move': np.array([1, 1]), 'attack': False},
    })
    # TODO: Add and tests rewards

    np.testing.assert_array_equal(obs['prey0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  1.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  1.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.allclose(obs['prey0']['resources'], np.array([
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,    0.47,  0.  ,  0.  ,  0.41,  0.36, -1.,  ],
        [-1.,   -1.,   -1.,    0.89,  0.38,  0.51,  0.28,  0.69, -1.,  ],
        [-1.,   -1.,   -1.,    0.9 ,  0.66,  0.49,  1.  ,  0.3 , -1.,  ],
        [-1.,   -1.,   -1.,    0.  ,  0.  ,  0.94,  0.  ,  0.  , -1.,  ],
        [-1.,   -1.,   -1.,    0.  ,  0.22,  1.  ,  0.  ,  0.98, -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ]
    ]))
    assert not done['prey0']
    
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  1.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  1.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    assert np.allclose(obs['prey1']['resources'], np.array([
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.47,  0.  ,  0.  ,  0.41,  0.36, -1.,   -1.,  ],
        [-1.,   -1.,    0.89,  0.38,  0.51,  0.28,  0.69, -1.,   -1.,  ],
        [-1.,   -1.,    0.9 ,  0.66,  0.49,  1.  ,  0.3 , -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.  ,  0.94,  0.  ,  0.  , -1.,   -1.,  ],
        [-1.,   -1.,    0.  ,  0.22,  1.  ,  0.  ,  0.98, -1.,   -1.,  ],
        [-1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,  ]
    ]))
    assert not done['prey1']

    np.testing.assert_array_equal(obs['predator0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  1.,  1.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.]
    ]))
    np.testing.assert_array_equal(obs['predator0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.]
    ]))
    assert not done['predator0']
    assert not done['__all__']


    obs, reward, done, _ = env.step({
        'prey0': {'move': np.array([0, 0]), 'harvest': 0.5},
        'prey1': {'move': np.array([0,  1]), 'harvest': 0.5},
        'predator0': {'move': np.array([0, 0]), 'attack': True},
    })

    np.testing.assert_array_equal(obs['prey0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  1.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  1.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    assert np.allclose(obs['prey0']['resources'], np.array([
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ],
        [-1.,   -1.,   -1.,    0.51,  0.  ,  0.  ,  0.45,  0.4 , -1.,  ],
        [-1.,   -1.,   -1.,    0.93,  0.  ,  0.01,  0.32,  0.73, -1.,  ],
        [-1.,   -1.,   -1.,    0.94,  0.7 ,  0.53,  1.  ,  0.34, -1.,  ],
        [-1.,   -1.,   -1.,    0.  ,  0.  ,  0.98,  0.  ,  0.  , -1.,  ],
        [-1.,   -1.,   -1.,    0.  ,  0.26,  1.  ,  0.  ,  1.  , -1.,  ],
        [-1.,   -1.,   -1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,  ]
    ]))
    assert done['prey0']

    np.testing.assert_array_equal(obs['prey1']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  1.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    assert np.allclose(obs['prey1']['resources'], np.array([
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.,  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.,  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.,  ],
        [-1.,    0.51,  0.  ,  0.  ,  0.45,  0.4 , -1.,   -1.,   -1.,  ],
        [-1.,    0.93,  0.  ,  0.01,  0.32,  0.73, -1.,   -1.,   -1.,  ],
        [-1.,    0.94,  0.7 ,  0.53,  1.  ,  0.34, -1.,   -1.,   -1.,  ],
        [-1.,    0.  ,  0.  ,  0.98,  0.  ,  0.  , -1.,   -1.,   -1.,  ],
        [-1.,    0.  ,  0.26,  1.  ,  0.  ,  1.  , -1.,   -1.,   -1.,  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.,  ],
    ]))
    assert not done['prey1']

    np.testing.assert_array_equal(obs['predator0']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.]
    ]))
    np.testing.assert_array_equal(obs['predator0']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  0.]
    ]))
    assert not done['predator0']
    assert not done['__all__']


    obs, reward, done, _ = env.step({
        'prey1': {'move': np.array([1,  0]), 'harvest': 0},
        'predator0': {'move': np.array([1, 1]), 'attack': False},
    })

    assert 'prey0' not in obs
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    assert np.allclose(obs['prey1']['resources'], np.array([
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,    0.55,  0.  ,  0.  ,  0.49,  0.44, -1.,   -1.,   -1.  ],
        [-1.,    0.97,  0.  ,  0.01,  0.36,  0.77, -1.,   -1.,   -1.  ],
        [-1.,    0.98,  0.74,  0.57,  1.  ,  0.38, -1.,   -1.,   -1.  ],
        [-1.,    0.  ,  0.  ,  1.  ,  0.  ,  0.  , -1.,   -1.,   -1.  ],
        [-1.,    0.  ,  0.3 ,  1.  ,  0.  ,  1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ]
    ]))
    assert not done['prey1']

    np.testing.assert_array_equal(obs['predator0']['position'][:,:,0], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    np.testing.assert_array_equal(obs['predator0']['position'][:,:,1], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    assert not done['predator0']
    assert not done['__all__']


    obs, reward, done, _ = env.step({
        'prey1': {'move': np.array([0,  0]), 'harvest': 0.5},
        'predator0': {'move': np.array([0, 0]), 'attack': True},
    })

    np.testing.assert_array_equal(obs['prey1']['position'][:,:,0], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    np.testing.assert_array_equal(obs['prey1']['position'][:,:,1], np.array([
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  1.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1., -1.]
    ]))
    assert np.allclose(obs['prey1']['resources'], np.array([
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,    0.59,  0.  ,  0.  ,  0.53,  0.48, -1.,   -1.,   -1.  ],
        [-1.,    1.  ,  0.  ,  0.01,  0.4 ,  0.81, -1.,   -1.,   -1.  ],
        [-1.,    1.  ,  0.78,  0.61,  0.54,  0.42, -1.,   -1.,   -1.  ],
        [-1.,    0.  ,  0.  ,  1.  ,  0.  ,  0.  , -1.,   -1.,   -1.  ],
        [-1.,    0.  ,  0.34,  1.  ,  0.  ,  1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ],
        [-1.,   -1.  , -1.  , -1.  , -1.  , -1.  , -1.,   -1.,   -1.  ]
    ]))
    assert done['prey1']

    np.testing.assert_array_equal(obs['predator0']['position'][:,:,0], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    np.testing.assert_array_equal(obs['predator0']['position'][:,:,1], np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]))
    assert not done['predator0']
    assert done['__all__']
