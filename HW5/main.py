import random
import gymnasium as gym
import flappy_bird_gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# --- Epsilon-Greedy Action Selection ---
def epsilon_greedy_action(model, state, epsilon):
    state_tensor = torch.tensor(state).float().unsqueeze(0)  # Convert state to tensor
    q_values = model(state_tensor)

    if random.random() < epsilon:
        return random.choice([0, 1])  # Random action (explore)
    else:
        return torch.argmax(q_values).item()  # Max Q-value action (exploit)


# --- Q-Learning Update ---
def train(model, replay_buffer, criterion, optimizer, batch_size, gamma):
    if replay_buffer.size() < batch_size:
        return  # Not enough data to train

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states).float()
    next_states = torch.tensor(next_states).float()
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones, dtype=torch.float32)  # Convert dones to float32 for arithmetic operations

    # Get Q-values for current states
    q_values = model(states)

    # Get Q-values for next states (max over actions)
    next_q_values = model(next_states)
    next_q_values_max = next_q_values.max(dim=1)[0]

    # Compute target Q-values
    targets = rewards + gamma * next_q_values_max * (1 - dones)

    # Select the Q-values corresponding to the actions taken
    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute loss
    loss = criterion(q_values_taken, targets)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- Preprocessing Lidar Data ---
def preprocess_observation_lidar(obs):
    # Normalize the lidar data (optional step based on your environment)
    # Ensure the data is within a reasonable range (e.g., 0 to 1)
    return np.array(obs)  # No additional processing required


# --- Q-Network (Neural Network) ---
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        # Define the layers for a simple fully connected network (MLP)
        self.fc1 = nn.Linear(180, 128)  # 180 inputs (Lidar data)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output 2 Q-values (for actions: 0 or 1)

    def forward(self, x):
        # Forward pass through the layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output 2 Q-values

        return x


# --- Main Training Loop ---
def train_agent():
    # Initialize the environment and model
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    model = QNetwork()
    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Hyperparameters
    epsilon = 1.0  # Initial epsilon for exploration
    epsilon_min = 0.05  # Minimum epsilon for exploration
    epsilon_decay = 0.995  # Epsilon decay factor
    gamma = 0.99  # Discount factor
    batch_size = 32  # Replay buffer batch size
    replay_buffer = ReplayBuffer(10000)  # Experience replay buffer
    target_update_freq = 1000  # Frequency of target network updates
    total_episodes = 1000  # Total training episodes
    max_timesteps = 500  # Maximum timesteps per episode
    skip_frame_count = 9  # Number of frames to skip after jumping

    # Reward shaping: penalize for being too close to the ceiling
    ceiling_threshold = 1.5 # Adjust this threshold based on your environment's height

    # Variable to track the maximum level reached
    max_level_reached = 0

    # Training loop
    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = preprocess_observation_lidar(obs)  # Preprocess lidar data
        done = False
        total_reward = 0
        frame_counter = 0  # Counter to reduce inference frequency
        skip_frames = 0  # Skip frames counter after jumping
        current_level = 0  # Track the current level for the episode
        last_level = current_level

        for t in range(max_timesteps):
            if skip_frames > 0:
                skip_frames -= 1  # Decrement the skip frame counter
                continue  # Skip this frame

            # Select action using epsilon-greedy policy
            action = epsilon_greedy_action(model, state, epsilon)

            # Take action in the environment
            next_obs, reward, terminated, _, info = env.step(action)
            next_state = preprocess_observation_lidar(next_obs)  # Preprocess next lidar observation

            # Track the current level (distance or score)
            current_level = info.get('score', 0)  # Replace with 'score' if available
            if last_level != current_level:
                reward += 25
                last_level = current_level

            # Check for ceiling proximity and penalize
            if next_obs[0] < ceiling_threshold:  # If close to ceiling, apply penalty
                reward -= 1  # Deduct reward for getting too close to ceiling

            # Store the experience in the replay buffer
            replay_buffer.push((state, action, reward, next_state, terminated))

            # Update the state
            state = next_state
            total_reward += reward

            # Train the model (reduce inference frequency to every few frames)
            frame_counter += 1
            if frame_counter % 10 == 0:  # Update every 4 frames, for example
                train(model, replay_buffer, criterion, optimizer, batch_size, gamma)

            # Skip frames after a jump action (action = 1)
            if action == 1:
                skip_frames = skip_frame_count  # Set the skip frames counter

            # If done, break out of the loop
            if terminated:
                break

        # Update maximum level reached if necessary
        if current_level > max_level_reached:
            max_level_reached = current_level

        # Decay epsilon more aggressively
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Print the episode stats: current level and maximum level reached
        print(f"Episode {episode + 1}/{total_episodes}, Total Reward: {total_reward}, "
              f"Current Level: {current_level}, Max Level Reached: {max_level_reached}, "
              f"Epsilon: {epsilon:.4f}")

    # Close the environment
    env.close()


train_agent()
