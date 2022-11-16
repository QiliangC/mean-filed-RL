import gym
from agent import MFQAgent, ILQAgent, MFACAgent
import numpy as np
from datetime import datetime
import time

if __name__ == "__main__":

    env = gym.make('ma_gym:Combat-v0')

    # agent = MFQAgent(state_size=env.observation_space[0].shape[0], action_size=10,
    #                        num_agent=env.n_agents, seed=1, learning_rate=1e-4, buffer_size=500000,
    #                        batch_size=128, update_every=1,step_update_network=1000, gamma=0.95,
    #                        beta=1, beta_decay=0, learning_mode=True)

    agent = MFACAgent(state_size=env.observation_space[0].shape[0], action_size=10,
                           num_agent=env.n_agents, seed=1,  buffer_size=500000,
                           batch_size=128, update_every=1,step_update_network=1000, gamma=0.95,
                           beta=1, beta_decay=0, learning_mode=True)

    # agent = ILQAgent(state_size=env.observation_space[0].shape[0], action_size=10,
    #                         num_agent=env.n_agents, seed=1, learning_rate=1e-4, buffer_size=500000,
    #                         batch_size=128, update_every=1, gamma=0.95, beta=1, beta_decay=2e-5, learning_mode=True)


    obs_n = env.reset()
    mean_action = np.ones((5, ))
    best_score = -np.inf
    total_reward_list = []
    save_path = "models/"
    max_epoch = 2000
    agent.load_model(save_path)
    for epoch in range(max_epoch):
        states = env.reset()
        total_reward = 0
        step = 0
        dones = [False for _ in range(env.n_agents)]
        while not all(dones):
            # env.render()
            # time.sleep(0.1)
            step += 1
            actions = agent.act(states, mean_action)
            mean_action = 1/env.n_agents * np.sum(actions) * np.ones((5,))
            next_states, rewards, dones, info = env.step(actions)
            # print(mean_action)
            total_reward += sum(rewards)

        total_reward_list.append(total_reward)
        avg_reward = np.mean(total_reward_list)

        # print(step)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if epoch % 1 == 0:
            print(current_time, "epoch:", epoch, "reward: ", total_reward,
                  "avg_reward: ", avg_reward, "beta:", agent.beta)
