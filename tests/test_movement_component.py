
from gym.spaces import Box
import numpy as np

from admiral.envs.components.state import GridPositionState, ContinuousPositionState, SpeedAngleState, VelocityState
from admiral.envs.components.actor import GridMovementActor, SpeedAngleMovementActor, AccelerationMovementActor
from admiral.envs.components.agent import PositionAgent, SpeedAngleAgent, SpeedAngleActingAgent, VelocityAgent, GridMovementAgent, AcceleratingAgent

class GridMovementTestAgent(GridMovementAgent, PositionAgent): pass

def test_grid_movement_component():
    agents = {
        'agent0': GridMovementTestAgent(id='agent0', initial_position=np.array([6, 4]), move_range=2),
        'agent1': GridMovementTestAgent(id='agent1', initial_position=np.array([3, 3]), move_range=3),
        'agent2': GridMovementTestAgent(id='agent2', initial_position=np.array([0, 1]), move_range=1),
        'agent3': GridMovementTestAgent(id='agent3', initial_position=np.array([8, 4]), move_range=1),
    }
    state = GridPositionState(region=10, agents=agents)
    actor = GridMovementActor(position_state=state, agents=agents)

    for agent in agents.values():
        assert 'move' in agent.action_space

    state.reset()

    np.testing.assert_array_equal(actor.process_move(agents['agent0'], np.array([-1,  1])), np.array([-1,  1]))
    np.testing.assert_array_equal(actor.process_move(agents['agent1'], np.array([ 0,  1])), np.array([ 0,  1]))
    np.testing.assert_array_equal(actor.process_move(agents['agent2'], np.array([ 0, -1])), np.array([ 0, -1]))
    np.testing.assert_array_equal(actor.process_move(agents['agent3'], np.array([ 1,  0])), np.array([ 1,  0]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([5, 5]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([3, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))

    np.testing.assert_array_equal(actor.process_move(agents['agent0'], np.array([ 2, -2])), np.array([ 2, -2]))
    np.testing.assert_array_equal(actor.process_move(agents['agent1'], np.array([-3,  0])), np.array([-3,  0]))
    np.testing.assert_array_equal(actor.process_move(agents['agent2'], np.array([-1,  0])), np.array([ 0,  0]))
    np.testing.assert_array_equal(actor.process_move(agents['agent3'], np.array([ 1,  1])), np.array([ 0,  0]))
    np.testing.assert_array_equal(agents['agent0'].position, np.array([7, 3]))
    np.testing.assert_array_equal(agents['agent1'].position, np.array([0, 4]))
    np.testing.assert_array_equal(agents['agent2'].position, np.array([0, 0]))
    np.testing.assert_array_equal(agents['agent3'].position, np.array([9, 4]))

class SpeedAngleMovementTestAgent(SpeedAngleAgent, SpeedAngleActingAgent, PositionAgent): pass

def test_speed_angle_movement_component():
    agents = {
        'agent0': SpeedAngleMovementTestAgent(id='agent0', initial_position=np.array([6.2,  3.3  ]), \
            initial_speed=1.0, min_speed=0.0, max_speed=1.0, max_acceleration=0.35, \
            initial_banking_angle=-30, max_banking_angle=45, max_banking_angle_change=30, \
            initial_ground_angle=300),
        'agent1': SpeedAngleMovementTestAgent(id='agent1', initial_position=np.array([2.1,  3.15 ]), \
            initial_speed=0.5, min_speed=0.0, max_speed=1.0, max_acceleration=0.35, \
            initial_banking_angle=10, max_banking_angle=45, max_banking_angle_change=30, \
            initial_ground_angle=100),
        'agent2': SpeedAngleMovementTestAgent(id='agent2', initial_position=np.array([0.5,  1.313]), \
            initial_speed=0.24, min_speed=0.0, max_speed=1.0, max_acceleration=0.35, \
            initial_banking_angle=0.0, max_banking_angle=45, max_banking_angle_change=30, \
            initial_ground_angle=45),
        'agent3': SpeedAngleMovementTestAgent(id='agent3', initial_position=np.array([8.24, 4.4  ]), \
            initial_speed=0.6, min_speed=0.0, max_speed=1.0, max_acceleration=0.35, \
            initial_banking_angle=24, max_banking_angle=45, max_banking_angle_change=30, \
            initial_ground_angle=180),
    }
    position_state = ContinuousPositionState(region=10, agents=agents)
    speed_angle_state = SpeedAngleState(agents=agents)
    actor = SpeedAngleMovementActor(position=position_state, speed_angle=speed_angle_state, agents=agents)

    for agent in agents.values():
        assert 'accelerate' in agent.action_space
        assert 'bank' in agent.action_space

    position_state.reset()
    speed_angle_state.reset()
    for agent in agents.values():
        np.testing.assert_array_equal(agent.position, agent.initial_position)
        assert agent.speed == agent.initial_speed
        assert agent.banking_angle == agent.initial_banking_angle
        assert agent.ground_angle == agent.initial_ground_angle
    
    assert np.allclose(actor.process_move(agents['agent0'], np.array([0.0]), np.array([0.0])), np.array([0.,          -1.]))
    assert np.allclose(agents['agent0'].position, np.array([6.2, 2.3]))
    assert np.allclose(actor.process_move(agents['agent1'], np.array([0.0]), np.array([0.0])), np.array([-0.17101007,  0.46984631]))
    assert np.allclose(agents['agent1'].position, np.array([1.92898993, 3.61984631]))
    assert np.allclose(actor.process_move(agents['agent2'], np.array([0.0]), np.array([0.0])), np.array([0.16970563,   0.16970563]))
    assert np.allclose(agents['agent2'].position, np.array([0.66970563, 1.48270563]))
    assert np.allclose(actor.process_move(agents['agent3'], np.array([0.0]), np.array([0.0])), np.array([-0.54812727, -0.24404199]))
    assert np.allclose(agents['agent3'].position, np.array([7.69187273, 4.15595801]))

    assert np.allclose(actor.process_move(agents['agent0'], np.array([-0.35]), np.array([30])), np.array([0,           -0.65      ]))
    assert np.allclose(agents['agent0'].position, np.array([6.2,  1.65]))
    assert np.allclose(actor.process_move(agents['agent1'], np.array([-0.1]), np.array([-30])), np.array([0,            0.4       ]))
    assert np.allclose(agents['agent1'].position, np.array([1.92898993, 4.01984631]))
    assert np.allclose(actor.process_move(agents['agent2'], np.array([-0.24]), np.array([30])), np.array([0,            0.0       ]))
    assert np.allclose(agents['agent2'].position, np.array([0.66970563, 1.48270563]))
    assert np.allclose(actor.process_move(agents['agent3'],  np.array([0.0]), np.array([-24])), np.array([-0.54812727, -0.24404199]))
    assert np.allclose(agents['agent3'].position, np.array([7.14374545, 3.91191603]))

class ParticleAgent(PositionAgent, VelocityAgent, AcceleratingAgent): pass

def test_acceleration_movement_component():

    agents = {
        'agent0': ParticleAgent(id='agent0', initial_position=np.array([2.3, 4.5]), initial_velocity=np.array([1, 0]), max_speed=1.0, max_acceleration=0.5),
        'agent1': ParticleAgent(id='agent1', initial_position=np.array([8.5, 1.0]), initial_velocity=np.array([0, 0]), max_speed=1.0, max_acceleration=0.5),
        'agent2': ParticleAgent(id='agent2', initial_position=np.array([5.0, 5.0]), initial_velocity=np.array([-1, -1]), max_speed=2.0, max_acceleration=0.5),
    }

    position_state = ContinuousPositionState(region=10, agents=agents)
    velocity_state = VelocityState(agents=agents, friction=0.1)
    actor = AccelerationMovementActor(position_state=position_state, velocity_state=velocity_state, agents=agents)

    for agent in agents.values():
        assert 'accelerate' in agent.action_space

    position_state.reset()
    velocity_state.reset()
    for agent in agents.values():
        np.testing.assert_array_equal(agent.position, agent.initial_position)
        np.testing.assert_array_equal(agent.velocity, agent.initial_velocity)

    
    assert np.allclose(actor.process_move(agents['agent0'], np.array([-1, 0])), np.array([0., 0.]))
    assert np.allclose(agents['agent0'].position, np.array([2.3, 4.5]))
    assert np.allclose(actor.process_move(agents['agent1'], np.array([-1, 1])), np.array([-0.70710678,  0.70710678]))
    assert np.allclose(agents['agent1'].position, np.array([7.79289322, 1.70710678]))
    assert np.allclose(actor.process_move(agents['agent2'], np.array([0, 0])), np.array([-1, -1]))
    assert np.allclose(agents['agent2'].position, np.array([4, 4]))

    velocity_state.apply_friction(agents['agent0'])
    assert np.allclose(agents['agent0'].velocity, np.array([0, 0]))
    velocity_state.apply_friction(agents['agent1'])
    assert np.allclose(agents['agent1'].velocity, np.array([-0.6363961, 0.6363961]))
