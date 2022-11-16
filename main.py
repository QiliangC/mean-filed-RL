import gym
from agent import MFQAgent, ILQAgent, MFACAgent
import numpy as np
import time
from datetime import datetime
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--testing_mode', default=False, action='store_true', help='decide learning mode or testing mode')
    parser.add_argument('--exploration_decay', type=float, default=2e-5, help='decide the exploration decay in IL and MFQ')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='decide the learning rate')
    parser.add_argument('--learning_rate_critic', type=float, default=1e-5, help='decide the learning rate')
    parser.add_argument('--learning_rate_actor', type=float, default=1e-6, help='decide the learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size when learning')
    parser.add_argument('--num_round', type=int, default=1000, help='set the trainning round')
    parser.add_argument('--seed', type=int, default=1, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')

    args = parser.parse_args()

    env = gym.make('ma_gym:Combat-v0')

    if args.algo == 'mfq':
        agent = MFQAgent(state_size=env.observation_space[0].shape[0], action_size=10,
                         num_agent=env.n_agents, seed=args.seed, learning_rate=args.learning_rate,
                         batch_size=128, beta_decay=args.exploration_decay, learning_mode=args.learning_mode)
    elif args.algo == 'il':
        agent = ILQAgent(state_size=env.observation_space[0].shape[0], action_size=10,
                         num_agent=env.n_agents, seed=args.seed, learning_rate=args.learning_rate,
                         batch_size=args.batch_size, gamma=args.gamma, beta_decay=args.exploration_decay,
                         learning_mode=args.learning_mode)
    else:
        agent = MFACAgent(state_size=env.observation_space[0].shape[0], action_size=10,
                          num_agent=env.n_agents, seed=args.seed, learning_rate_critic=args.learning_rate_critic,
                          learning_rate_actor=args.learning_rate_actor, batch_size=args.batch_size,
                          gamma=args.gamma, learning_mode=True)

    save_path = "models/" + args.algo
    if not args.testing_mode:
        agent.load_model(save_path)
    # agent.load_model(save_path)

    obs_n = env.reset()
    mean_action = np.ones((5, ))
    best_score = -np.inf
    total_reward_list = []

    for epoch in range(args.num_round):
        states = env.reset()
        total_reward = 0
        step = 0
        dones = [False for _ in range(env.n_agents)]
        while not all(dones):
            if args.render:
                env.render()
                time.sleep(0.1)
            step += 1
            actions = agent.act(states, mean_action)
            mean_action = 1/env.n_agents * np.sum(actions) * np.ones((5,))
            next_states, rewards, dones, info = env.step(actions)
            # print(mean_action)
            total_reward += sum(rewards)
            if args.testing_mode:
                agent.store_memory(states, actions, rewards, next_states, mean_action, dones)
                agent.learn()
        total_reward_list.append(total_reward)
        avg_reward = np.mean(total_reward_list[-100:])
        if avg_reward > best_score and epoch > 100:
            best_score = avg_reward
            if args.testing_mode:
                agent.save_model(save_path)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if epoch % 10 == 0:
            print(current_time, "epoch:", epoch, "avg_reward: ", avg_reward,
                  "best_score: ", best_score)
    np.save('record/total_reward_curve_' + args.algo + '_seed_' + str(args.seed) + '_epoch_' + str(args.num_round), total_reward_list)