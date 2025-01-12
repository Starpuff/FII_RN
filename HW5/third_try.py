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
        self.capacity = capacity

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    def save(self):
        """Serialize the buffer as a list for saving."""
        return list(self.buffer)

    def load(self, saved_buffer):
        """Load a saved buffer from a list."""
        self.buffer = deque(saved_buffer, maxlen=self.capacity)



# --- Epsilon-Greedy Action Selection ---
def epsilon_greedy_action(model, state, epsilon):
    state_tensor = torch.tensor(state).float().unsqueeze(0)  # Convert state to tensor
    q_values = model(state_tensor)

    if random.random() < epsilon:
        return random.choice([0, 1])  # Random action (explore)
    else:
        return torch.argmax(q_values).item()  # Max Q-value action (exploit)


# --- Q-Learning Update ---
def train(model, target_model, replay_buffer, criterion, optimizer, batch_size, gamma):
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

    # Get Q-values for next states using the target network
    next_q_values = target_model(next_states)
    next_q_values_max = next_q_values.max(dim=1)[0]

    # Compute target Q-values
    targets = rewards + gamma * next_q_values_max * (1 - dones)

    # Select the Q-values corresponding to the actions taken
    q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute loss
    loss = criterion(q_values_taken, targets.detach())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- Preprocessing Lidar Data ---
def preprocess_observation_lidar(obs):
    return np.clip(np.array(obs) / 10.0, 0, 1)  # Normalize lidar data to [0, 1]


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


# --- Save Checkpoint ---
def save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, max_level_reached, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': replay_buffer.save(),  # Serialize the buffer
        'episode': episode,
        'epsilon': epsilon,
        'max_level_reached': max_level_reached,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename} (overwritten)")


# --- Load Checkpoint ---
def load_checkpoint(model, optimizer, replay_buffer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    replay_buffer.load(checkpoint['replay_buffer'])  # Deserialize the buffer
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    max_level_reached = checkpoint['max_level_reached']
    print(f"Checkpoint loaded from {filename}")
    return episode, epsilon, max_level_reached



# --- Main Training Loop ---
def train_agent():
    # Initialize the environment and model
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    model = QNetwork()
    target_model = QNetwork()
    target_model.load_state_dict(model.state_dict())  # Synchronize with model
    target_model.eval()  # Set target model to evaluation mode

    criterion = nn.SmoothL1Loss()  # Huber Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Hyperparameters
    epsilon = 1.0  # Initial epsilon for exploration
    epsilon_min = 0.05  # Minimum epsilon for exploration
    epsilon_decay = 0.999  # Epsilon decay factor
    gamma = 0.99  # Discount factor
    batch_size = 32  # Replay buffer batch size
    replay_buffer = ReplayBuffer(10000)  # Experience replay buffer
    target_update_freq = 100  # Frequency of target network updates
    total_episodes = 1000  # Total training episodes
    max_timesteps = 500  # Maximum timesteps per episode

    # Checkpoint filename
    checkpoint_filename = "save_file33333.txt"

    # # Try loading checkpoint if it exists
    # try:
    #     episode, epsilon, max_level_reached = load_checkpoint(model, optimizer, replay_buffer, checkpoint_filename)
    # except FileNotFoundError:
    #     print("No checkpoint found, starting fresh.")
    #     episode = 0
    #     max_level_reached = 0

    episode = 0
    max_level_reached = 0
    # Training loop
    for episode in range(episode, total_episodes):
        obs, _ = env.reset()
        state = preprocess_observation_lidar(obs)  # Preprocess lidar data
        done = False
        total_reward = 0

        for t in range(max_timesteps):
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_action(model, state, epsilon)

            # Take action in the environment
            next_obs, reward, terminated, _, info = env.step(action)
            next_state = preprocess_observation_lidar(next_obs)  # Preprocess next lidar observation

            # Store the experience in the replay buffer
            replay_buffer.push((state, action, reward, next_state, terminated))

            # Update the state
            state = next_state
            total_reward += reward

            # Train the model
            train(model, target_model, replay_buffer, criterion, optimizer, batch_size, gamma)

            # If done, break out of the loop
            if terminated:
                break

        # Update the target network periodically
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Save the checkpoint after each episode
        save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, max_level_reached, checkpoint_filename)

        # Print episode stats
        print(f"Episode {episode + 1}/{total_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Close the environment
    env.close()


train_agent()
