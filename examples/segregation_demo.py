
from amber.envs.segregation import Segregation

# Policy
tolerance = 0.3
def compute_action(obs):
    if obs[0] + obs[1] == 0: # there are no neighbors
        return 1 # Will move because wants 0.3 same and has 0 same
    return 0 if float(obs[0]) / float(obs[0] + obs[1]) >= tolerance else 1

# Environment
env = Segregation.build()
obs = env.reset()
env.render()
for i in range(100):
    action = {agent_id: compute_action(agent_obs) for agent_id, agent_obs in obs.items()}
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
        break

from matplotlib import pyplot as plt
plt.show()