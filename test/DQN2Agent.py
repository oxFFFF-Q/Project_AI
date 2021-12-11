from pommerman.agents.simple_agent import SimpleAgent
import torch
import torch.nn as nn
import torch.optim as optim
from pommerman import characters
import torch.nn.functional as F

import random
import gym
import numpy as np
from utils import featurize, featurize2

from pommerman.agents import BaseAgent
from replay_buffer import ReplayBuffer, ReplayBuffer2


class DQN2Agent(BaseAgent):
    """
    DQN from scratch
    """
    def __init__(self, env, args, character=characters.Bomber):
        super(DQN2Agent, self).__init__(character)
        self.obs_n = env.observation_space.shape[0]  # output_dim
        self.action_n = 6  # input_dim
        self.env = env

        self.epsilon = args.epsilon
        self.eps_decay = args.eps_decay
        self.min_eps = args.min_eps
        self.gamma = args.gamma
        self.lr = args.lr
        self.baseAgent = SimpleAgent()
        self.episodes = args.episodes
        self.maxsteps = args.maxsteps
        self.showevery = args.showevery

        self.capacity = args.capacity
        self.batch = args.batch
        # self.buffer = ReplayBuffer(self.capacity, self.batch)
        self.buffer = ReplayBuffer2(self.capacity)
        """神经网络"""
        self.model = nn.Sequential(
            nn.Linear(199, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_n)
        )
        
        self.learn_step_counter = 0
        self.eval_net, self.target_net = Net2(self.env), Net2(self.env)
        #self.optim = optim.Adam(self.model.parameters(), self.lr) #adam优化算法
        self.optim = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.MSE_loss = nn.MSELoss()

    def forward(self, state):
        state = torch.FloatTensor(state)
        qvals = self.model(state)
        return qvals

    def act(self, obs, action_space):
        return self.baseAgent.act(obs,self.action_n)

    def dqnact(self, obs):
        action = self.eval_net.forward(obs)[0]
        result = (torch.max(action, 0)[1]).numpy()
        return result

    def update(self, gamma, batch_size):
        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        states, actions, rewards, next_states, done = self.buffer.sample2(batch_size)
        action_index = actions.squeeze(-2)[:,0].unsqueeze(1)
        curr_Q_batch = self.eval_net(states)[:,0]
        curr_Q = curr_Q_batch.gather(1, action_index).squeeze(-1)
        #print(curr_Q)
        
        next_batch = self.target_net(next_states)[:,0]
        next_Q = torch.max(next_batch,1)[0]
        #print(next_Q)

        rewards_batch = rewards.squeeze(-2)[:,0]
        #print(rewards_batch)
        # expected_Q = rewards + self.gamma * torch.max(next_Q, 1)
        expected_Q = gamma * next_Q + rewards_batch

        # max_q_prime = next_Q.max(1)[0].unsqueeze(1)
        # expected_Q = done * (rewards + gamma * max_q_prime) + (1 - done) * 1 / (1 - gamma) * rewards
        # expected_Q = done * (rewards + gamma * max_q_prime) + 1 / (1 - gamma) * rewards
        loss = self.MSE_loss(curr_Q, expected_Q) # TODO: try Huber Loss later too

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def epsdecay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_eps else self.epsilon


class Net2(nn.Module):
    def __init__(self,env):
        super(Net2,self).__init__()
        self.obs_n = env.observation_space.shape[0]
        self.fc1 = nn.Linear(374,128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128,6)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.fc1(x)
        x = F.relu(x)
        qvals = self.out(x)
        return qvals