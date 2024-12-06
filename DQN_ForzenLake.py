import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple
import random
import math
import logging
import matplotlib.pyplot as plt



class DQN(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)
    

class DQNFrozenLake:
    def __init__(self, grid_size = 4, gamma = 0.99, lr = 1e-3):
        self.env = gym.make("FrozenLake-v1", map_name=f"{grid_size}x{grid_size}", is_slippery=True)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # DQN hyperparameters
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1000
        self.steps_done = 0
        self.learning_rate = lr
        self.epsilon = self.epsilon_start

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.Transition = namedtuple('Transition', 
                                   ('state', 'action', 'reward', 'next_state', 'terminated'))

    def state_to_tensor(self, state):
        state_tensor = torch.zeros(self.state_size)
        state_tensor[state] = 1
        return state_tensor.to(self.device)
    
    def select_action(self, state):
        rand_value = random.random()
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        # Exploration
        if rand_value < self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        else:
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
            
    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append(self.Transition(state, action, reward, next_state, terminated))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))

        # Recorded states of simulations
        state_batch = torch.stack([self.state_to_tensor(s) for s in batch.state])
        # Recorded actions of simulations
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s) for s in batch.next_state])
        terminated_batch = torch.tensor(batch.terminated, device=self.device, dtype=torch.float)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute next Q values for calulating expected Q values, based on Bellman Equation
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            # For terminal state, no future state, then the reward is only immediate reward
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - terminated_batch)

        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, episodes = 1000, target_update = 10, record_video = False):
        rewards_history = []

        if record_video:
            env = gym.make(self.env.unwrapped.spec.id, render_mode='rgb_array',
                          map_name="4x4", is_slippery=True)
            env = gym.wrappers.RecordVideo(env, "dqn_frozen_lake_videos",
                                         episode_trigger=lambda x: x % 100 == 0)
        else:
            env = self.env
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            terminated = False

            while not terminated:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                terminated = terminated or truncated

                self.remember(state, action, reward, next_state, terminated)
                loss = self.replay()
                state = next_state
                total_reward += reward

            if episode % target_update == 0:
                self.update_target_network()

            rewards_history.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                logging.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}")

        env.close()
        return rewards_history
    
    def save_model(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)


    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = DQNFrozenLake()
    rewards = agent.train(episodes=1000, record_video=True)
    agent.save_model("dqn_frozen_lake.pth")

    plt.figure(figsize=(10,6))
    plt.plot(rewards)    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.3f}")



            

    



