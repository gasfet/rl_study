# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, node_num):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, node_num),
            nn.ReLU(),
            nn.Linear(node_num, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, node_num),
            nn.ReLU(),
            nn.Linear(node_num, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.node_num = 12
        self.learning_rate = 0.002
        self.discount_rate = 0.95
        self.penalty = -100
        self.episode_num = 500
        self.moving_avg_size = 20

        self.model = ActorCritic(self.state_size, self.action_size, self.node_num).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.reward_list = []
        self.count_list = []
        self.moving_avg_list = []

    def train(self):
        for episode in range(self.episode_num):
            state = self.env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            count, reward_tot = self.run_episode(state)

            if count < 500:
                reward_tot += self.penalty

            self.reward_list.append(reward_tot)
            self.count_list.append(count)
            self.moving_avg_list.append(self.moving_avg(self.count_list, self.moving_avg_size))

            if episode % 10 == 0:
                print(f"Episode: {episode}, Moving Avg: {self.moving_avg_list[-1]}, Rewards Avg: {np.mean(self.reward_list)}")

        self.save_model()

    def run_episode(self, state):
        reward_tot = 0
        count = 0
        done = False

        while not done:
            count += 1
            action_prob, _ = self.model(state)
            action = torch.multinomial(action_prob, 1).item()

            next_state, reward, done, _, _ = self.env.step(action)
            reward_tot += reward

            if done and count < 500:
                reward = self.penalty

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            self.train_step(state, next_state, reward, action, done)
            state = next_state

        return count, reward_tot

    def train_step(self, state, next_state, reward, action, done):
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        action_prob, state_value = self.model(state)
        _, next_state_value = self.model(next_state)

        # Compute advantages and targets
        if done:
            advantage = reward - state_value
            target = reward
        else:
            advantage = reward + self.discount_rate * next_state_value - state_value
            target = reward + self.discount_rate * next_state_value

        # Actor loss
        log_prob = torch.log(action_prob.squeeze(0)[action])
        actor_loss = -log_prob * advantage.detach()

        # Critic loss
        critic_loss = nn.functional.mse_loss(state_value, target.detach())

        # Backpropagation
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def moving_avg(self, data, size=10):
        if len(data) > size:
            return np.mean(data[-size:])
        return np.mean(data)

    def save_model(self):
        torch.save(self.model.state_dict(), "./model/a2c.pth")
        print("***** End A2C Learning *****")


if __name__ == "__main__":
    agent = Agent()
    agent.train()
