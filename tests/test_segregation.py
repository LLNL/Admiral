
from gym.spaces import MultiBinary, Box
import numpy as np
import networkx as nx
import pytest

from amber.envs.segregation import Segregation, RedAgent, GreenAgent

def test_builder():
    env = Segregation.build()
    assert env.region == 10
    assert env.neighbors == 8

    assert len(env.agents) == 74
    for agent in env.agents.values():
        assert agent.observation_space == Box(low=0, high=8, shape=(3,), dtype=np.int)
        assert agent.action_space == MultiBinary(1)

def test_builder_region():
    env = Segregation.build({'region': 5})
    assert env.region == 5
    assert len(env.agents) == 18

def test_builder_neighbors():
    env = Segregation.build({'neighbors': 4})
    assert env.neighbors == 4

def test_builder_agents_list():
    agents = {
        'red0': RedAgent('red0'),
        'red1': RedAgent('red1'),
        'green0': GreenAgent('green0'),
        'green1': GreenAgent('green1'),
        'green2': GreenAgent('green2')
    }
    env = Segregation.build({'agents': agents})
    assert len(env.agents) == 5

def test_reset_and_step():
    np.random.seed(24)
    env = Segregation.build({'region': 5})

    obs = env.reset()
    assert env.grid.nodes[(0,0)] == {}
    assert env.grid.nodes[(0,1)] == {}
    assert env.grid.nodes[(0,2)] == {}
    assert env.grid.nodes[(0,3)] == {}
    assert env.grid.nodes[(0,4)] == {}
    assert isinstance(env.grid.nodes[(1,0)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(1,1)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(1,2)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(1,3)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(1,4)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(2,0)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(2,1)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(2,2)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(2,3)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(2,4)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(3,0)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(3,1)]['agent'], RedAgent) 
    assert env.grid.nodes[(3,2)] == {}
    assert isinstance(env.grid.nodes[(3,3)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(3,4)]['agent'], GreenAgent) 
    assert isinstance(env.grid.nodes[(4,0)]['agent'], GreenAgent) 
    assert env.grid.nodes[(4,1)] == {}
    assert isinstance(env.grid.nodes[(4,2)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(4,3)]['agent'], RedAgent) 
    assert isinstance(env.grid.nodes[(4,4)]['agent'], GreenAgent)
    
    np.testing.assert_array_equal(obs['green0'], np.array([3, 2, 3]))
    np.testing.assert_array_equal(obs['green1'], np.array([2, 1, 2]))
    np.testing.assert_array_equal(obs['green2'], np.array([1, 4, 3]))
    np.testing.assert_array_equal(obs['green3'], np.array([1, 4, 0]))
    np.testing.assert_array_equal(obs['green4'], np.array([3, 2, 0]))
    np.testing.assert_array_equal(obs['green5'], np.array([2, 5, 1]))
    np.testing.assert_array_equal(obs['green6'], np.array([1, 2, 0]))
    np.testing.assert_array_equal(obs['green7'], np.array([1, 1, 1]))
    np.testing.assert_array_equal(obs['green8'], np.array([2, 2, 1]))
    np.testing.assert_array_equal(obs['red0'], np.array([2, 4, 2]))
    np.testing.assert_array_equal(obs['red1'], np.array([3, 0, 2]))
    np.testing.assert_array_equal(obs['red2'], np.array([2, 1, 2]))
    np.testing.assert_array_equal(obs['red3'], np.array([4, 3, 1]))
    np.testing.assert_array_equal(obs['red4'], np.array([2, 3, 3]))
    np.testing.assert_array_equal(obs['red5'], np.array([3, 2, 0]))
    np.testing.assert_array_equal(obs['red6'], np.array([2, 5, 1]))
    np.testing.assert_array_equal(obs['red7'], np.array([2, 2, 1]))
    np.testing.assert_array_equal(obs['red8'], np.array([4, 3, 1]))


    action = {agent.id: 0 for agent in env.agents.values()}
    action['red4'] = 1
    obs, reward, done, info = env.step(action)
    assert done == False


    action = {agent.id: 0 for agent in env.agents.values()}
    obs, reward, done, info = env.step(action)
    assert done == True