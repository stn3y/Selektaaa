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
        self.memory = deque(maxlen=10000)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = 1.0
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = config.get('batch_size', 64)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if random.random() <= self.epsilon:
            selected_action = random.choice(available_actions)
            logging.info(f"Random action selected: {selected_action}")
            return selected_action
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        act_values = self.model(state)
        
        sorted_actions = torch.argsort(act_values, descending=True).cpu().numpy()[0]
        for action in sorted_actions:
            if action in available_actions:
                logging.info(f"Model action selected: {action}")
                return action
        
        # Fallback in case the model's best actions are not available
        fallback_action = random.choice(available_actions)
        logging.warning(f"Fallback action selected: {fallback_action}")
        return fallback_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            logging.warning("Not enough samples in memory to replay.")
            return
        minibatch = random.sample(self.memory, self.batch_size)

        logging.info("Training on a new batch...")

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            target_f = self.model(state).detach()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        logging.info("Batch training complete.")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logging.info(f"Epsilon decayed to {self.epsilon:.4f}")

        # Soft update of the target network's weights
        self.update_target_model()

    def update_target_model(self):
        # Soft update of the target network's weights
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.1 * param.data + 0.9 * target_param.data)

        logging.info("Target network updated.")

    def generate_playlist(self, env, context_based=False, user_context=None):
        logging.info("Starting playlist generation...")
        state = env.reset()

        if context_based and user_context is not None:
            logging.info("Using context-aware playlist generation.")
            playlist = env.generate_context_aware_playlist(user_context)
        else:
            playlist = []
            while not env.done:
                action = self.act(state, env.available_tracks)
                if action is None:
                    break
                next_state, reward, done, available_tracks = env.step(action)
                playlist.append(env.tracks[action])
                self.remember(state, action, reward, next_state, done)
                state = next_state

        logging.info("Playlist generation complete.")
        return playlist

    def train(self):
        self.replay()