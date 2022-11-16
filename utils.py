import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer():
    def __init__(self, max_mem, input_dims, num_agents):
        self.max_mem = max_mem
        self.obs_mem = np.zeros([max_mem, num_agents, input_dims], dtype=np.float32)
        self.act_mem = np.zeros([max_mem, num_agents], dtype=np.int64)
        self.reward_mem = np.zeros([max_mem, num_agents], dtype=np.float32)
        self.new_obs_mem = np.zeros([max_mem, num_agents, input_dims], dtype=np.float32)
        self.mean_act_mem = np.zeros([max_mem, num_agents], dtype=np.int64)
        self.termination = np.zeros([max_mem, num_agents], dtype=np.bool)
        self.mem_cnt = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def store_transition(self, observation, action, reward, observation_, mean_action, done):
        index = self.mem_cnt % self.max_mem
        self.obs_mem[index] = observation
        self.act_mem[index] = action
        self.reward_mem[index] = reward
        self.new_obs_mem[index] = observation_
        self.mean_act_mem[index] = mean_action
        self.termination[index] = done
        self.mem_cnt += 1

    def sample_transition(self, batch_size):
        max_size = min(self.max_mem, self.mem_cnt)
        index = np.random.choice(max_size, batch_size, replace=False)
        observations = torch.from_numpy(self.obs_mem[index]).to(self.device)
        actions = torch.from_numpy(self.act_mem[index]).to(self.device)
        rewards = torch.from_numpy(self.reward_mem[index]).to(self.device)
        observations_ = torch.from_numpy(self.new_obs_mem[index]).to(self.device)
        mean_actions = torch.from_numpy(self.mean_act_mem[index]).to(self.device)
        dones = torch.from_numpy(self.termination[index]).to(self.device)

        return observations, actions, rewards, observations_, mean_actions, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return self.mem_cnt

def Boltzmann_policy(Q_list, beta):
    Q_list = np.array(Q_list)
    denominator = np.sum(np.exp(-beta * Q_list))
    policy_prob = np.exp(-beta * Q_list) / denominator

    return policy_prob

class MFQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, name, file_path='models/', fc1_units=256, fc2_units=256):  # 64, 64
        super(MFQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.save_path = os.path.join(file_path, name)
        self.name = name
        self.fc1 = nn.Linear(state_size + 1, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, mean_action):
        """Build a network that maps state -> action values."""
        state_mean_action = torch.concat((state, mean_action), dim=1)

        x = F.relu(self.fc1(state_mean_action))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_name = os.path.join(save_path, self.name)
        torch.save(self.state_dict(), save_path_name)

    def load_model(self, save_path):
        save_path_name = os.path.join(save_path, self.name)
        self.load_state_dict(torch.load(save_path_name, map_location='cpu'))

class MFActorNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, name, file_path='models/', fc1_units=256, fc2_units=256):  # 64, 64
        super(MFActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.save_path = os.path.join(file_path, name)
        self.name = name
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_name = os.path.join(save_path, self.name)
        torch.save(self.state_dict(), save_path_name)

    def load_model(self, save_path):
        save_path_name = os.path.join(save_path, self.name)
        self.load_state_dict(torch.load(save_path_name, map_location='cpu'))


class ILQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, name, file_path='models/', fc1_units=256, fc2_units=256):  # 64, 64
        super(ILQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.save_path = os.path.join(file_path, name)
        self.name = name
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_name = os.path.join(save_path, self.name)
        torch.save(self.state_dict(), save_path_name)

    def load_model(self, save_path):
        save_path_name = os.path.join(save_path, self.name)
        self.load_state_dict(torch.load(save_path_name, map_location='cpu'))
