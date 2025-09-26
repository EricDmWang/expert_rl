from gridworld25_env import GridWorld25v0
import numpy as np

env = GridWorld25v0(mode="mode_2",seed=1)
obs,info = env.reset()
env.render()
for i in range(10):
    actions = [env.action_space.sample() for _ in range(4)]
    obs, rewards, terminated, truncated, info = env.step(actions)
    env.render()
    if terminated or truncated:
        break
obs,info = env.reset()    
env.close()

