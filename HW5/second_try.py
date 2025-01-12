import random
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def epsilon_greedy_action(model, state, epsilon):
    state_tensor = torch.tensor(state).float().unsqueeze(0)
    q_values = model(state_tensor)

    if random.random() < epsilon:
        return random.choices([0, 1], weights=[9, 1])[0]
    else:
        return torch.argmax(q_values).item()


def train(model, replay_buffer, criterion, optimizer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states).float()
    next_states = torch.tensor(next_states).float()
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states)

    next_q_values = model(next_states)
    next_q_values_max = next_q_values.max(dim=1)[0]

    targets = rewards + gamma * next_q_values_max * (1 - dones)

    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    loss = criterion(q_values_taken, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def preprocess_observation_lidar(obs):
    return np.array(obs)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, max_level_reached, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer,
        'episode': episode,
        'epsilon': epsilon,
        'max_level_reached': max_level_reached,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename} (overwritten)")


def load_checkpoint(model, optimizer, replay_buffer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    replay_buffer.buffer = checkpoint['replay_buffer']
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    max_level_reached = checkpoint['max_level_reached']
    print(f"Checkpoint loaded from {filename}")
    return episode, epsilon, max_level_reached


def train_agent():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    model = QNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 32
    replay_buffer = ReplayBuffer(10000)
    target_update_freq = 1000
    total_episodes = 1000
    max_timesteps = 500
    skip_frame_count = 9

    ceiling_threshold = 1.15

    max_level_reached = 0

    checkpoint_filename = "save_file222222.txt"

    # Try loading checkpoint if it exists
    # try:
    #     episode, epsilon, max_level_reached = load_checkpoint(model, optimizer, replay_buffer, checkpoint_filename)
    # except FileNotFoundError:
    #     print("No checkpoint found, starting fresh.")

    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = preprocess_observation_lidar(obs)
        done = False
        total_reward = 0
        frame_counter = 0
        skip_frames = 2
        current_level = 0
        last_level = current_level
        non_flap_counter = 0
        non_flap_reward_threshold = 10

        for t in range(max_timesteps):
            if skip_frames > 0:
                skip_frames -= 1
                continue

            action = epsilon_greedy_action(model, state, epsilon)

            next_obs, reward, terminated, _, info = env.step(action)
            next_state = preprocess_observation_lidar(next_obs)

            current_level = info.get('score', 0)
            if last_level != current_level:
                reward += 200
                last_level = current_level

            if next_obs[0] < ceiling_threshold:
                reward -= 1

            if action == 0:
                non_flap_counter += 1
                if non_flap_counter % non_flap_reward_threshold == 0:
                    reward += 50
            else:
                non_flap_counter = 0

            replay_buffer.push((state, action, reward, next_state, terminated))

            state = next_state
            total_reward += reward

            frame_counter += 1
            if frame_counter % 4 == 0:
                train(model, replay_buffer, criterion, optimizer, batch_size, gamma)

            if action == 1:
                skip_frames = skip_frame_count

            if terminated:
                break

        if current_level > max_level_reached:
            max_level_reached = current_level

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{total_episodes}, Total Reward: {total_reward}, "
              f"Current Level: {current_level}, Max Level Reached: {max_level_reached}, "
              f"Epsilon: {epsilon:.4f}")

        save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, max_level_reached, checkpoint_filename)

    env.close()


train_agent()
