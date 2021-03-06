
from admiral.tools import utils as adu

def _get_checkpoint(full_trained_directory, checkpoint_desired):
    """
    Return the checkpoint directory to load the policy. If checkpoint_desired is specified and
    found, then return that policy. Otherwise, return the last policy.
    """
    checkpoint_dirs = adu.find_dirs_in_dir('checkpoint*', full_trained_directory)

    # Try to load the desired checkpoint
    if checkpoint_desired is not None: # checkpoint specified
        for checkpoint in checkpoint_dirs:
            if checkpoint_desired == int(checkpoint.split('/')[-1].split('_')[-1]):
                return checkpoint, checkpoint_desired
        import warnings
        warnings.warn('Could not find checkpoint_{}. Attempting to load the last checkpoint.'.format(checkpoint_desired))
    
    # Load the last checkpoint
    max_checkpoint = None
    max_checkpoint_value = 0
    for checkpoint in checkpoint_dirs:
        checkpoint_value = int(checkpoint.split('/')[-1].split('_')[-1])
        if checkpoint_value > max_checkpoint_value:
            max_checkpoint_value = checkpoint_value
            max_checkpoint = checkpoint
    
    if max_checkpoint is None:
        raise FileNotFoundError("Did not find a checkpoint file in the given directory.")
    
    return max_checkpoint, max_checkpoint_value

def run(full_trained_directory, parameters):
    """Play MARL policies from a saved policy"""

    # Load the experiment as a module
    # First, we must find the .py file in the directory
    import os
    py_files = [file for file in os.listdir(full_trained_directory) if file.endswith('.py')]
    assert len(py_files) == 1
    full_path_to_config = os.path.join(full_trained_directory, py_files[0])
    experiment_mod = adu.custom_import_module(full_path_to_config)
    
    checkpoint_dir, checkpoint_value = _get_checkpoint(full_trained_directory, parameters.checkpoint)
    print(checkpoint_dir)

    # Play with ray
    import ray
    import ray.rllib
    ray.init()

    # Get the agent
    alg = ray.rllib.agents.registry.get_agent_class(experiment_mod.params['ray_tune']['run_or_experiment'])
    agent = alg(
        env=experiment_mod.params['ray_tune']['config']['env'],
        config=experiment_mod.params['ray_tune']['config']    
    )
    agent.restore(os.path.join(checkpoint_dir, 'checkpoint-' + str(checkpoint_value)))

    # Render the environment
    from ray.rllib.env import MultiAgentEnv
    env = agent.workers.local_worker().env

    # Determine if we are single- or multi-agent case.
    def _multi_get_action(obs, done=None, env=None, policy_agent_mapping=None, **kwargs):
        joint_action = {}
        if done is None:
            done = {agent: False for agent in obs}
        for agent_id, agent_obs in obs.items():
            if done[agent_id]: continue # Don't get actions for done agents
            policy_id = policy_agent_mapping(agent_id)
            action = agent.compute_action(agent_obs, policy_id=policy_id, \
                explore=parameters.no_explore)
            joint_action[agent_id] = action
        return joint_action
    
    def _single_get_action(obs, agent=None, **kwargs):
        return agent.compute_action(obs, explore=parameters.no_explore)

    def _multi_get_done(done):
        return done['__all__']
    
    def _single_get_done(done):
        return done
    
    policy_agent_mapping = None
    if isinstance(env, MultiAgentEnv):
        policy_agent_mapping = agent.config['multiagent']['policy_mapping_fn']
        _get_action = _multi_get_action
        _get_done = _multi_get_done
    else:
        _get_action = _single_get_action
        _get_done = _single_get_done

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    for episode in range(parameters.episodes):
        print('Episode: {}'.format(episode))
        obs = env.reset()
        done = None
        all_done = False
        fig = plt.figure()

        def gen_frame_until_done():
            nonlocal all_done
            i = 0
            while not all_done:
                i += 1
                yield i

        def animate(i):
            nonlocal obs, done
            env.render(fig=fig)
            plt.pause(1e-16)
            action = _get_action(obs, done=done, env=env, agent=agent, policy_agent_mapping=policy_agent_mapping)
            obs, _, done, _ = env.step(action)
            if _get_done(done):
                nonlocal all_done
                all_done = True
                env.render(fig=fig)
                plt.pause(1e-16)
                plt.close(fig)

        anim = FuncAnimation(fig, animate, frames=gen_frame_until_done, repeat=False, \
            interval=parameters.frame_delay)
        if parameters.record:
            anim.save(os.path.join(full_trained_directory, 'Episode_{}.mp4'.format(episode)))
        plt.show(block=False)
        while all_done is False:
            plt.pause(1)

    ray.shutdown()