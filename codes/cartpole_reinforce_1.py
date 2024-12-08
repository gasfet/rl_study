import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random as rand
import matplotlib.pyplot as plt

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self, state_size, action_size, node_num):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(state_size, node_num)
        self.fc2 = nn.Linear(node_num, action_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, states):
        x = torch.tanh(self.fc1(states))
        return self.softmax(self.fc2(x))


class Agent():

    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.value_size = 1

        self.node_num = 12
        self.learning_rate = 0.0005
        self.epochs_cnt = 5
        self.model = MyModel(self.state_size, self.action_size, self.node_num).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.model = self.build_model().to(device)

        self.discount_rate = 0.95
        self.penalty = -10
        self.episode_num = 500
        self.moving_avg_size = 20

        self.reward_list = []
        self.count_list = []
        self.moving_avg_list = []

        self.states, self.action_matrixs, self.action_probs, self.rewards = [], [], [], []

        # self.DUMMY_ACTION_MATRIX, self.DUMMY_REWARD = np.zeros((1, 1, self.action_size)), np.zeros((1, 1, self.value_size))


    def train(self):
        for episode in range(self.episode_num):
            state, _ = self.env.reset()
            self.env.max_episode_steps = 500

            count, reward_tot = self.make_memory(episode, state)
            self.train_mini_batch()
            self.clear_memory()

            if count < 500:
                reward_tot += self.penalty

            self.reward_list.append(reward_tot)
            self.count_list.append(count)
            self.moving_avg_list.append(self.moving_avg(self.count_list, self.moving_avg_size))

            if (episode % 10 == 0):
                print("episode:{}, moving_avg:{}, rewards_avg:{}".format(episode, self.moving_avg_list[-1], np.mean(self.reward_list)))

        self.save_model()


    def make_memory(self, episode, state):
        reward_tot = 0
        count = 0
        reward = np.zeros(self.value_size)
        action_matrix = np.zeros(self.action_size)
        done = False
        while not done:
            count += 1

            state_t = np.reshape(state, [1, self.state_size])
            action_matrix_t = np.reshape(action_matrix, [1, self.action_size])

            state_t = torch.tensor(state_t, dtype=torch.float32).to(device)
            action_matrix_t = torch.tensor(action_matrix_t, dtype=torch.float32).to(device)

            # model forward pass
            with torch.no_grad():
                action_prob = self.model(state_t).cpu().numpy()

            action = np.random.choice(self.action_size, 1, p=action_prob[0])[0]
            action_matrix = np.zeros(self.action_size)
            action_matrix[action] = 1

            state_next, reward, done, none, none2 = self.env.step(action)

            if count < 500 and done:
                reward = self.penalty

            # self.states.append(np.reshape(state_t.cpu().numpy(), [1, self.state_size]))
            # self.action_matrixs.append(np.reshape(action_matrix, [1, self.action_size]))
            # self.action_probs.append(np.reshape(action_prob, [1, self.action_size]))
            self.states.append(state_t.cpu().numpy())
            self.action_matrixs.append(action_matrix)
            self.action_probs.append(action_prob)
            self.rewards.append(reward)

            reward_tot += reward
            state = state_next
        return count, reward_tot


    def clear_memory(self):
        self.states, self.action_matrixs, self.action_probs, self.rewards = [], [], [], []  # clear memory

    def make_discount_rewards(self, rewards):
        discounted_rewards = np.zeros(np.array(rewards).shape)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_rate + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards


    def train_mini_batch(self):
        discount_rewards = self.make_discount_rewards(self.rewards)
        discount_rewards_t = torch.tensor(discount_rewards, dtype=torch.float32).to(device).unsqueeze(1)

        states_t = torch.tensor(np.vstack(self.states), dtype=torch.float32).to(device)
        action_matrixs_t = torch.tensor(np.vstack(self.action_matrixs), dtype=torch.float32).to(device)

        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(states_t)
        action_probs = torch.sum(action_matrixs_t * y_pred, dim=-1)
        loss = -torch.log(action_probs) * discount_rewards_t

        loss = loss.mean()
        loss.backward()
        self.optimizer.step()



    def moving_avg(self, data, size=10):
        if len(data) > size:
            c = np.array(data[-size:])
        else:
            c = np.array(data)
        return np.mean(c)

    def save_model(self):
        torch.save(self.model.state_dict(), "./model/reinforce1.pth")
        print("*****End Learning")


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train()
    
    # Plot rewards and moving averages
    plt.figure(figsize=(10, 5))
    plt.plot(agent.reward_list, label="Rewards")
    plt.plot(agent.moving_avg_list, linewidth=4, label="Moving Average")
    plt.legend(loc="upper left")
    plt.title("REINFORCE")
    plt.show()

if __name__ == "__main__":
    main()

