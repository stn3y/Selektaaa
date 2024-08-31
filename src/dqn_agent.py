import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.advantage = nn.Linear(128, output_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        advantage = self.advantage(x)
        value = self.value(x)

        return value + advantage - advantage.mean()

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(device)
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        self.gamma = config['gamma']
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.learning_rate = config['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.tau = config.get('tau', 0.1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if random.random() <= self.epsilon:
            return random.choice(available_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        valid_actions = [(i, val) for i, val in enumerate(act_values[0]) if i in available_actions]
        valid_actions.sort(key=lambda x: x[1], reverse=True)

        if not valid_actions:
            return random.choice(available_actions)
        return valid_actions[0][0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_f = self.model(state)

            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def generate_playlist(self, env):
        state = env.reset()
        playlist = []
        while not env.done:
            action = self.act(state, env.available_tracks)
            if action is None:
                break
            next_state, reward, done, available_tracks = env.step(action)
            playlist.append(env.tracks[action])
            self.remember(state, action, reward, next_state, done)
            state = next_state

        return playlist