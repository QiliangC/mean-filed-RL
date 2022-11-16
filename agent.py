import numpy as np
import random

import torch
import torch.distributions
import torch.nn.functional as F
import torch.optim as optim

from utils import ReplayBuffer, MFQNetwork, ILQNetwork, MFActorNetwork, Boltzmann_policy

class ILQAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agent, seed, learning_rate=1e-4, buffer_size=500000, batch_size=128
                 , gamma=0.95, beta=1, beta_decay=1e-4, beta_min=0.01, tau=0.95, learning_mode=True):
        """Initialize an Agent object. """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.beta = beta
        self.beta_dacay = beta_decay
        self.beta_min = beta_min
        self.gamma = gamma
        self.tau = tau
        self.num_agent = num_agent
        self.seed = random.seed(seed)
        self.learning = learning_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Q-Network
        self.qnetwork_local = [ILQNetwork(self.state_size, action_size, seed, name="Q_local_" + str(i)).to(self.device)
                               for i in range(num_agent)]
        self.qnetwork_target = [ILQNetwork(self.state_size, action_size, seed, name="Q_target_" + str(i)).to(self.device)
            for i in range(num_agent)]
        self.optimizer = [optim.Adam(x.parameters(), lr=learning_rate) for x in self.qnetwork_local]

        # Replay memory
        self.memory = ReplayBuffer(max_mem=buffer_size, input_dims=state_size, num_agents=self.num_agent)
        self.t_step = 0

    def store_memory(self, state, action, reward, next_state, mean_action, done):
        self.memory.store_transition(state, action, reward, next_state, mean_action, done)


    def act(self, state, mean_action):
        """Returns actions for given state as per current policy."""
        action_values = []
        action_list = []
        state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        for i in range(self.num_agent):
            self.qnetwork_local[i].eval()
            with torch.no_grad():
                action_value = self.qnetwork_local[i](state[:, i])
                action_values.append(action_value)
            self.qnetwork_local[i].train()
            if self.learning:
                if random.random() > self.beta:
                    action = np.argmax(action_values[i].cpu().detach().numpy())
                else:
                    action = random.choice(np.arange(self.action_size))
                action_list.append(action)
            else:
                action = np.argmax(action_values[i].cpu().detach().numpy())
                action_list.append(action)
        return action_list

    def decrement_beta(self):
        self.beta = self.beta - self.beta_dacay if self.beta > self.beta_min \
            else self.beta_min

    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, mean_actions, dones = \
                self.memory.sample_transition(self.batch_size)
        else:
            return
        """Update value parameters using given batch of experience tuples. """
        # Get max predicted Q values (for next states) from target model
        for i in range(self.num_agent):
            indice = np.arange(self.batch_size)
            Q_pred = self.qnetwork_local[i](states[:,i])[indice, actions[:,i]]
            Q_next = self.qnetwork_target[i](next_states[:,i]).max(dim=1)[0]
            Q_next[dones[:,i]] = 0.0
            target = rewards[:,i] + self.gamma * Q_next
            loss = F.mse_loss(target, Q_pred)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()

        self.decrement_beta()
        # ------------------- update target network ------------------- #
        for i in range(self.num_agent):
            self.soft_update(self.qnetwork_local[i], self.qnetwork_target[i], self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filepath):
        print("...save model...")
        for i in range(self.num_agent):
            self.qnetwork_local[i].save_model(filepath)
            self.qnetwork_target[i].save_model(filepath)

    def load_model(self, filepath):
        print("...load model...")
        for i in range(self.num_agent):
            self.qnetwork_local[i].load_model(filepath)
            self.qnetwork_target[i].load_model(filepath)


class MFQAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agent, seed, learning_rate=1e-4, buffer_size=500000, batch_size=128
                 , step_update_network=1000, gamma=0.95, beta=1, beta_decay=1e-4, beta_min=0.01, tau=0.95, learning_mode=True):
        """Initialize an Agent object. """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.step_update_network = step_update_network
        self.beta = beta
        self.beta_dacay = beta_decay
        self.beta_min = beta_min
        self.gamma = gamma
        self.tau = tau
        self.num_agent = num_agent
        self.seed = random.seed(seed)
        self.learning = learning_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Q-Network
        self.qnetwork_local = [MFQNetwork(self.state_size, action_size, seed, name="Q_local_" + str(i)).to(self.device)
                               for i in range(num_agent)]
        self.qnetwork_target = [MFQNetwork(self.state_size, action_size, seed, name="Q_target_" + str(i)).to(self.device)
            for i in range(num_agent)]
        self.optimizer = [optim.Adam(x.parameters(), lr=learning_rate) for x in self.qnetwork_local]

        # Replay memory
        self.memory = ReplayBuffer(max_mem=buffer_size, input_dims=state_size, num_agents=self.num_agent)
        self.t_step = 0

    def store_memory(self, state, action, reward, next_state, mean_action, done):
        self.memory.store_transition(state, action, reward, next_state, mean_action, done)

    def act(self, state, mean_action):
        """Returns actions for given state as per current policy."""
        action_values = []
        action_list = []
        state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean_action = np.asarray(mean_action)
        mean_action = torch.from_numpy(mean_action).float().unsqueeze(1).unsqueeze(1).to(self.device)
        for i in range(self.num_agent):
            self.qnetwork_local[i].eval()
            with torch.no_grad():
                action_value = self.qnetwork_local[i](state[:, i], mean_action[i])
                action_values.append(action_value)
            self.qnetwork_local[i].train()
            if self.learning:
                action_prob = Boltzmann_policy(action_values[i].cpu(), self.beta)
                action = np.random.choice(self.action_size, p=action_prob[0])
                action_list.append(action)
            else:
                action = np.argmax(action_values[i].cpu().detach().numpy())
                action_list.append(action)

        return action_list

    def decrement_beta(self):
        self.beta = self.beta - self.beta_dacay if self.beta > self.beta_min \
            else self.beta_min

    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, mean_actions, dones = \
                self.memory.sample_transition(self.batch_size)
        else:
            return
        """Update value parameters using given batch of experience tuples. """
        # Get max predicted Q values (for next states) from target model
        for i in range(self.num_agent):
            Q_targets_next = self.qnetwork_target[i](next_states[:,i], mean_actions[:,i].unsqueeze(1))
            action_prob_next = Boltzmann_policy(Q_targets_next.cpu().detach().numpy(), beta=0)
            Value_next = np.sum(np.multiply(Q_targets_next[:,].cpu().detach().numpy(), action_prob_next[:,]),axis=1)

            Q_targets = rewards[:,i].cpu() + self.gamma * Value_next * (1 - dones[:,i].cpu().detach().numpy())

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local[i](states[:,i], mean_actions[:,i].unsqueeze(1)).gather(1, actions).to(self.device)

            Q_target_expand = torch.zeros([128, 5], dtype=torch.float32).to(self.device)


            for j in range(5):
                Q_target_expand[:,j] = Q_targets

            # Compute loss
            loss = F.mse_loss(Q_expected, Q_target_expand)
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
        self.decrement_beta()
        self.t_step += 1

        # ------------------- update target network ------------------- #
        if self.t_step % self.step_update_network == 0:
            for i in range(self.num_agent):
                self.soft_update(self.qnetwork_local[i], self.qnetwork_target[i], self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filepath):
        print("...save model...")
        for i in range(self.num_agent):
            self.qnetwork_local[i].save_model(filepath)
            self.qnetwork_target[i].save_model(filepath)

    def load_model(self, filepath):
        print("...load model...")
        for i in range(self.num_agent):
            self.qnetwork_local[i].load_model(filepath)
            self.qnetwork_target[i].load_model(filepath)

class MFACAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agent, seed, learning_rate_critic=1e-5,learning_rate_actor=1e-6, buffer_size=500000, batch_size=128
                 , step_update_network=1000, gamma=0.95, tau=0.95, learning_mode=True):
        """Initialize an Agent object. """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.step_update_network = step_update_network
        self.gamma = gamma
        self.tau = tau
        self.num_agent = num_agent
        self.seed = random.seed(seed)
        self.learning = learning_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Q-Network
        self.critic_local = [MFQNetwork(self.state_size, action_size, seed, name="Q_local_" + str(i)).to(self.device)
                               for i in range(num_agent)]
        self.critic_target = [MFQNetwork(self.state_size, action_size, seed, name="Q_target_" + str(i)).to(self.device)
            for i in range(num_agent)]

        self.actor_local = [MFActorNetwork(self.state_size, action_size, seed, name="Actor_local_" + str(i)).to(self.device)
                             for i in range(num_agent)]
        self.actor_target = [MFActorNetwork(self.state_size, action_size, seed, name="Actor_target_" + str(i)).to(self.device)
                              for i in range(num_agent)]

        self.optimizer_critic = [optim.Adam(x.parameters(), lr=learning_rate_critic) for x in self.critic_local]
        self.optimizer_actor = [optim.Adam(x.parameters(), lr=learning_rate_actor) for x in self.actor_local]


        # Replay memory
        self.memory = ReplayBuffer(max_mem=buffer_size, input_dims=state_size, num_agents=self.num_agent)
        self.t_step = 0

    def store_memory(self, state, action, reward, next_state, mean_action, done):
        self.memory.store_transition(state, action, reward, next_state, mean_action, done)

    def act(self, state, mean_action):
        """Returns actions for given state as per current policy."""
        action_values = []
        action_list = []
        state = np.asarray(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        for i in range(self.num_agent):
            self.actor_local[i].eval()
            with torch.no_grad():
                action_value = self.actor_local[i](state[:, i])
                action_values.append(action_value)
            self.actor_local[i].train()
            if self.learning:
                action_prob = F.softmax(action_value, dim=1)
                action_probs = torch.distributions.Categorical(action_prob)
                action = action_probs.sample().cpu().detach().numpy()
                action_list.append(action[0])
            else:
                action_prob = F.softmax(action_value, dim=1)
                action_probs = torch.distributions.Categorical(action_prob)
                action = action_probs.sample().cpu().detach().numpy()
                action_list.append(action)
        return action_list

    def decrement_beta(self):
        self.beta = self.beta - self.beta_dacay if self.beta > self.beta_min \
            else self.beta_min

    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, mean_actions, dones = \
                self.memory.sample_transition(self.batch_size)
        else:
            return
        """Update value parameters using given batch of experience tuples. """
        for i in range(self.num_agent):
            Q_targets_next = self.critic_target[i](next_states[:,i], mean_actions[:,i].unsqueeze(1))
            action_prob_next = self.actor_target[i](next_states[:,i])
            Value_next = torch.sum(torch.bmm(Q_targets_next.view(self.batch_size, 1, -1), action_prob_next.view(self.batch_size, -1, 1)))
            Q_targets = rewards[:,i] + self.gamma * Value_next * (~dones[:,i])

            # Get expected Q values from local model
            Q_expected = self.critic_local[i](states[:,i], mean_actions[:,i].unsqueeze(1)).gather(1, actions)

            Q_target_expand = torch.zeros([128, 5], dtype=torch.float32).to(self.device)

            for j in range(5):
                Q_target_expand[:,j] = Q_targets
            # Compute loss
            critic_loss = F.mse_loss(Q_expected, Q_target_expand) / self.batch_size
            action_value = self.actor_local[i](next_states[:, i])
            action_prob = F.softmax(action_value, dim=1)

            log_probs = torch.log(action_prob)
            actor_loss = -torch.sum(torch.bmm(log_probs.gather(1, actions[:,i].unsqueeze(1)).view(self.batch_size, 1, -1), Q_targets_next.gather(1, actions[:,i].unsqueeze(1)).view(self.batch_size, -1, 1))) / self.batch_size

            loss = critic_loss + actor_loss
            self.optimizer_critic[i].zero_grad()
            self.optimizer_actor[i].zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local[i].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critic_local[i].parameters(), 1)

            self.optimizer_critic[i].step()
            self.optimizer_actor[i].step()

        self.t_step += 1

        # ------------------- update target network ------------------- #
        if self.t_step % self.step_update_network == 0:
            for i in range(self.num_agent):
                self.soft_update(self.critic_local[i], self.critic_target[i], self.tau)
                self.soft_update(self.actor_local[i], self.actor_target[i], self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filepath):
        print("...save model...")
        for i in range(self.num_agent):
            self.critic_local[i].save_model(filepath)
            self.critic_target[i].save_model(filepath)
            self.actor_local[i].save_model(filepath)
            self.actor_target[i].save_model(filepath)
    def load_model(self, filepath):
        print("...load model...")
        for i in range(self.num_agent):
            self.critic_local[i].load_model(filepath)
            self.critic_target[i].load_model(filepath)
            self.actor_local[i].save_model(filepath)
            self.actor_target[i].save_model(filepath)