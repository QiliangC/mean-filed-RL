import gym
import time
import numpy as np

env = gym.make('ma_gym:Combat-v0')

obs_n = env.reset()

step = 0
reward_list = []
for i in range(10000):
    ep_reward = 0
    # obs_n = env.reset()
    done_n = [False for _ in range(env.n_agents)]
    while not all(done_n):
        step += 1
        # env.render()
        # time.sleep(0.1)
        obs_n, reward_n, done_n, info = env.step(env.action_space.sample())

        ep_reward += sum(reward_n)
    reward_list.append(ep_reward)
    env.close()
    # print(step)
    print("epoch:", i, "avg_reward:", np.mean(reward_list))