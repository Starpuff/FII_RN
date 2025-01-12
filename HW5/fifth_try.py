import os
import time
import random
import itertools
from collections import deque

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# Constants
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Neural Network for DQN
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_dueling=True):
        super(DuelingQNetwork, self).__init__()
        self.use_dueling = use_dueling

        self.shared_layer = nn.Linear(state_dim, hidden_dim)

        if use_dueling:
            # Dueling Q-network components
            self.value_layer = nn.Linear(hidden_dim, hidden_dim)
            self.value_output = nn.Linear(hidden_dim, 1)
            self.advantage_layer = nn.Linear(hidden_dim, hidden_dim)
            self.advantage_output = nn.Linear(hidden_dim, action_dim)
        else:
            self.q_output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.shared_layer(x))

        if self.use_dueling:
            value = F.relu(self.value_layer(x))
            value = self.value_output(value)
            advantage = F.relu(self.advantage_layer(x))
            advantage = self.advantage_output(advantage)
            q_values = value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        else:
            q_values = self.q_output(x)

        return q_values

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self):
        # Hyperparameters
        self.environment_id = 'FlappyBird-v0'
        self.learning_rate = 0.0001
        self.discount_rate = 0.99
        self.target_update_frequency = 10
        self.replay_buffer_capacity = 100000
        self.batch_size = 32
        self.initial_exploration_rate = 1.0
        self.exploration_decay_rate = 0.9999995
        self.minimum_exploration_rate = 0.05
        self.max_reward_threshold = 100000
        self.hidden_layer_size = 512
        self.enable_double_dqn = True

        # Paths
        self.log_filepath = os.path.join(RUNS_DIR, 'flappybird.log')
        self.model_filepath = os.path.join(RUNS_DIR, 'flappybird.pt')

        # Neural Network
        self.loss_function = nn.MSELoss()
        self.optimizer = None

    def run(self, training_mode=True, render_environment=False):
        self._log_message(f"{'Training' if training_mode else 'Evaluation'} session started...")

        env = gym.make(self.environment_id, render_mode='human' if render_environment else None)
        action_space_size = env.action_space.n
        state_space_size = env.observation_space.shape[0]

        policy_network = DuelingQNetwork(state_space_size, action_space_size, self.hidden_layer_size, self.enable_double_dqn).to(device)

        if training_mode:
            self._train_agent(env, policy_network, state_space_size, action_space_size)
        else:
            policy_network.load_state_dict(torch.load(self.model_filepath))
            policy_network.eval()
            self._evaluate_agent(env, policy_network)

    def _train_agent(self, env, policy_network, state_space_size, action_space_size):
        exploration_rate = self.initial_exploration_rate
        replay_buffer = ReplayBuffer(self.replay_buffer_capacity)
        target_network = DuelingQNetwork(state_space_size, action_space_size, self.hidden_layer_size, self.enable_double_dqn).to(device)
        target_network.load_state_dict(policy_network.state_dict())
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=self.learning_rate)

        episode_rewards = []
        highest_reward = float('-inf')
        step_counter = 0
        session_start_time = time.time()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            done = False
            episode_total_reward = 0.0

            while not done and episode_total_reward < self.max_reward_threshold:
                action = self._choose_action(env, policy_network, state, exploration_rate, training=True)
                next_state, reward, done, _, _ = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                replay_buffer.add_experience((state, action, next_state, reward, done))
                state = next_state
                episode_total_reward += reward
                step_counter += 1

                if len(replay_buffer) > self.batch_size:
                    batch = replay_buffer.sample_batch(self.batch_size)
                    self._optimize_model(batch, policy_network, target_network)

                    exploration_rate = max(exploration_rate * self.exploration_decay_rate, self.minimum_exploration_rate)

                    if step_counter >= self.target_update_frequency:
                        target_network.load_state_dict(policy_network.state_dict())
                        step_counter = 0

            episode_rewards.append(episode_total_reward)
            if episode_total_reward > highest_reward:
                highest_reward = episode_total_reward
                torch.save(policy_network.state_dict(), self.model_filepath)
                self._log_message(f"New highest reward {episode_total_reward:.1f} achieved at episode {episode}")

    def _evaluate_agent(self, env, policy_network):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        done = False
        total_reward = 0.0

        while not done:
            action = self._choose_action(env, policy_network, state, exploration_rate=0, training=False)
            state, reward, done, _, _ = env.step(action.item())
            state = torch.tensor(state, dtype=torch.float, device=device)
            total_reward += reward

        print(f"Evaluation completed. Total reward: {total_reward}")

    def _choose_action(self, env, policy_network, state, exploration_rate, training):
        if training and random.random() < exploration_rate:
            return torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
        with torch.no_grad():
            return policy_network(state.unsqueeze(0)).squeeze().argmax()

    def _optimize_model(self, batch, policy_network, target_network):
        states, actions, next_states, rewards, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones, dtype=torch.float, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_next_actions = policy_network(next_states).argmax(dim=1)
                target_q_values = rewards + (1 - dones) * self.discount_rate * \
                                 target_network(next_states).gather(1, best_next_actions.unsqueeze(1)).squeeze()
            else:
                target_q_values = rewards + (1 - dones) * self.discount_rate * target_network(next_states).max(1)[0]

        current_q_values = policy_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _log_message(self, message):
        print(message)
        with open(self.log_filepath, 'a') as log_file:
            log_file.write(f"{time.time()}: {message}\n")

# Main Execution
if __name__ == "__main__":
    is_training_mode = True
    agent = DQNAgent()
    agent.run(training_mode=is_training_mode, render_environment=not is_training_mode)
