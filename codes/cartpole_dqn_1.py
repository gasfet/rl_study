﻿import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random as rand
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, node_num):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, node_num),
            nn.ReLU(),
            nn.Linear(node_num, action_size)
        )

    def forward(self, state):
        return self.net(state)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.node_num = 12
        self.learning_rate = 0.001
        self.epochs_cnt = 5

        self.model = DQN(self.state_size, self.action_size, self.node_num).to(device)
        # 역할: 네트워크 구조를 정의하고, 입력 데이터를 처리하여 예측값을 생성합니다.
        # 필요한 이유: 모델은 입력 데이터를 학습 가능한 가중치와 결합하여 결과를 예측하는 주요 계산 그래프입니다.

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #역할: 모델의 가중치를 업데이트합니다. 손실 함수의 기울기를 계산한 후 최적화 알고리즘을 사용해 가중치를 조정합니다.
        #필요한 이유: 네트워크가 손실을 줄이면서 더 나은 성능으로 학습할 수 있게 만듭니다.

        self.criterion = nn.MSELoss()
        # 역할: 예측값과 실제값 간의 차이를 계산하는 손실 함수를 정의합니다.
        # 필요한 이유: 손실 함수는 네트워크가 얼마나 잘 학습하고 있는지를 측정하는 기준입니다.

        self.discount_rate = 0.97
        self.penalty = -100

        self.episode_num = 1000
        
        self.replay_memory_limit = 2048
        self.replay_size = 32
        self.replay_memory = []

        self.epsilon = 0.99
        self.epsilon_decay = 0.2
        self.epsilon_min = 0.05

        self.moving_avg_size = 20
        self.reward_list = []
        self.count_list = []
        self.moving_avg_list = []

    def train(self):
        for episode in range(self.episode_num):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            Q, count, reward_tot = self.take_action_and_append_memory(episode, state)

            if count < 500:
                reward_tot += self.penalty

            self.reward_list.append(reward_tot)
            self.count_list.append(count)
            self.moving_avg_list.append(self.moving_avg(self.count_list, self.moving_avg_size))

            self.train_mini_batch()

            if episode % 10 == 0:
                print(f"Episode: {episode}, Moving Avg: {self.moving_avg_list[-1]:.2f}, Rewards Avg: {np.mean(self.reward_list):.2f}")

        self.save_model()

    def take_action_and_append_memory(self, episode, state):
        reward_tot = 0
        count = 0
        done = False
        epsilon = self.get_epsilon(episode)

        while not done:
            count += 1
            with torch.no_grad():
                Q = self.model(state)
            action = self.greedy_search(epsilon, Q)

            next_state, reward, done, _, _ = self.env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                reward += self.penalty

            self.replay_memory.append([state, action, reward, next_state, done])
            if len(self.replay_memory) > self.replay_memory_limit:
                self.replay_memory.pop(0)

            reward_tot += reward
            state = next_state

        return Q, count, reward_tot

    def train_mini_batch(self):
        if len(self.replay_memory) < self.replay_size:
            return

        samples = rand.sample(self.replay_memory, self.replay_size)
        batch_states = torch.cat([s[0] for s in samples])
        batch_actions = torch.tensor([s[1] for s in samples], device=device, dtype=torch.long)
        batch_rewards = torch.tensor([s[2] for s in samples], device=device, dtype=torch.float32)
        batch_next_states = torch.cat([s[3] for s in samples])
        batch_dones = torch.tensor([s[4] for s in samples], device=device, dtype=torch.float32)

        # Predict Q values for current and next states
        Q_values = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            Q_next = self.model(batch_next_states).max(1)[0]
            Q_target = batch_rewards + (1 - batch_dones) * self.discount_rate * Q_next

        loss = self.criterion(Q_values, Q_target)        
        
        self.optimizer.zero_grad() # clear gradients
        loss.backward()
        self.optimizer.step()

    def get_epsilon(self, episode):
        result = self.epsilon * (1 - episode / (self.episode_num * self.epsilon_decay))
        return max(result, self.epsilon_min)

    def greedy_search(self, epsilon, Q):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return torch.argmax(Q).item()

    def moving_avg(self, data, size=10):
        if len(data) > size:
            return np.mean(data[-size:])
        return np.mean(data)

    def save_model(self):
        torch.save(self.model.state_dict(), "./model/dqn.pth")
        print("***** End Training *****")


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train()
    
    # Plot rewards and moving averages
    plt.figure(figsize=(10, 5))
    plt.plot(agent.reward_list, label="Rewards")
    plt.plot(agent.moving_avg_list, linewidth=4, label="Moving Average")
    plt.legend(loc="upper left")
    plt.title("DQN")
    plt.show()

if __name__ == "__main__":
    main()
    
