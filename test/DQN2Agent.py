from pommerman.agents.simple_agent import SimpleAgent
from pommerman.agents.random_agent import RandomAgent
import torch
import torch.nn as nn
import torch.optim as optim
from pommerman import characters
import torch.nn.functional as F

import random
import gym
import numpy as np
from utils import featurize, featurize2
import os

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
        self.action_space = [0,1,2,3,4,5]
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
        return random.randrange(0,6,1)
        #return self.baseAgent.act(obs,self.action_n)

    def dqnact(self, obs):
        #action = self.eval_net.forward(obs)[0]
        #result = (torch.max(action, 0)[1]).numpy()
        lx = obs['local']
        ax = obs['additional']
        action = self.eval_net.forward1(lx,ax)[0]
        result = (torch.max(action, 0)[1]).numpy()
        return result

    def reward(self, featurel, featurea, action, sl, sa, rewards):
        # set up reward
        r_wood = 0.05
        r_powerup = 0.2
        r_put_bomb = 0.08
        r_win = 0.5
        r_fail = -1
        r_kick = 0.2
        r_hit = -0.2
        r_stay_in_danger = -0.2
        r_escape = 0.2

        rigid = featurel[0].numpy()
        wood = featurel[1].numpy()
        bomb = featurel[2].numpy()
        power_up = featurel[3]
        fog = featurel[4]
        agent1 = featurel[5].numpy()
        sagent1 = sl[5].numpy()
        agent2 = featurel[6].numpy()
        agent3 = featurel[7].numpy()
        agent4 = featurel[8].numpy()
        flame = featurel[9]
        position0 = int(featurea[0].item())
        position1 = int(featurea[1].item())
        p0 = int(sa[0].item())
        p1 = int(sa[1].item())
        ammo = int(sa[2].item())
        blast_strength = int(featurea[3].item())
        can_kick = int(featurea[4].item())
        teammate = int(featurea[5].item())
        enemies = int(featurea[6].item())
        message = int(featurea[7].item())
        rewards = rewards.numpy()
        reward = 0
        #sagents = sl[4]
        sbomb = sl[2].numpy()
        action = int(action.item())



        # print(agent1)
        # print(p0, p1)
        # print(position0, position1)
        # print('len:', len(agent1))
        # print('shape:', agent1.shape)
        # print('-------------------')
        # # reward_done
        # # print(rewards)

        if rewards == 1:
            reward += r_win
        if rewards == -1:
            reward += r_fail

        # reward_powerup
        # sammo = int(sa[2].item())
        # if ammo > 1 and ammo > sammo:
        #     reward += r_powerup
        # sstrength = int(sa[3].item())
        # if blast_strength > sstrength:
        #     reward += r_powerup
        # skick = int(sa[4].item())
        # if can_kick and not skick:
        #     reward += r_powerup

        # reward_wood
        if ammo > 0 and action == 5:
            bomb_flame = self.build_flame(position0, position1, rigid, blast_strength)
            num_wood = np.count_nonzero(wood*bomb_flame == 1)
            reward += num_wood*r_wood

        # reward_kick
        # if sbomb[position0, position1] == 1 and rewards != -1:
        #     reward += r_kick
        # reward_hit_wood
        if action>0 and action<5:
            if (p0,p1) == (position0,position1):
                reward += r_hit
        
        # reward_escape
        bomb_list = []
        for i in range(11):
            for j in range(11):
                if sbomb[i,j] == 1:
                    bomb_list.append((i,j))
        
        for b in bomb_list:
            bomb_flame1 = self.build_flame(b[0], b[1], rigid, 8)
            if bomb_flame1[position0, position1] == 1:
                reward += r_stay_in_danger
                #print(bomb_flame1)

            
        
        # reward escape from bomb

        """
        exist_bomb = []
        for row, rowbomb in enumerate(bomb):
            for col, _ in enumerate(rowbomb):
                if bomb[row, col] == 1:
                    exist_bomb.append((row, col))
        #print(bomb)
        #print(exist_bomb)
        
        if exist_bomb:
            for ebomb in exist_bomb:
                bomb_flame1 = self.build_flame(ebomb[0], ebomb[1], rigid, blast_strength)
                if bomb_flame1[position0, position1] == 1:
                    reward -= 0.5
                #print(bomb_flame1)
        """
        return reward

    def build_flame(self, position0, position1, rigid, blast_strength):
        
        position_bomb = np.array([position0,position1])
        m = position_bomb[0]
        n = position_bomb[1]
        l = blast_strength
        f = [l,l,l,l]       # Scope of flame: up down left right
        bomb_flame = np.zeros_like(rigid)

        # 判断实体墙或边界是否阻断火焰
        flame_up = np.zeros_like(bomb_flame)
        flame_down = np.zeros_like(bomb_flame)
        flame_left = np.zeros_like(bomb_flame)
        flame_right = np.zeros_like(bomb_flame)
        if m - f[0] < 0:  # 上边界
            f[0] = m
        flame_up[m - f[0]:m, n] = 1
        if m + f[1] > bomb_flame.shape[0] - 1:  # 下边界
            f[1] = bomb_flame.shape[0] - 1 - m
        flame_down[m + 1:m + f[1] + 1, n] = 1
        if n - f[2] < 0:  # 左边界
            f[2] = n
        flame_left[m, n - f[2]:n] = 1
        if n + f[3] > bomb_flame.shape[0] - 1:  # 右边界
            f[3] = bomb_flame.shape[0] - 1 - n
        flame_right[m, n + 1:n + f[3] + 1] = 1

        rigid_0 = flame_up * rigid
        rigid_1 = flame_down * rigid
        rigid_2 = flame_left * rigid
        rigid_3 = flame_right * rigid
        if np.argwhere(rigid_0==1).size != 0:    # 上实体墙
            rigid_up = np.max(np.argwhere(rigid_0==1)[:,0][0])
            if rigid_up >= m-f[0]:
                f[0] = m - rigid_up - 1
        if np.argwhere(rigid_1==1).size != 0:   # 下实体墙
            rigid_down = np.min(np.argwhere(rigid_1 == 1)[:, 0][0])
            if rigid_down <= m+f[1]:
                f[1] = rigid_down - m - 1
        if np.argwhere(rigid_2==1).size != 0:  # 左实体墙
            rigid_left = np.max(np.argwhere(rigid_2 == 1)[0, :][1])
            if rigid_left >= n-f[2]:
                f[2] = n - rigid_left - 1
        if np.argwhere(rigid_3==1).size != 0:  # 右实体墙
            rigid_right = np.min(np.argwhere(rigid_3 == 1)[0, :][1])
            if rigid_right <= n+f[3]:
                f[3] = rigid_right - n - 1
        bomb_flame[m-f[0]:m+f[1]+1, n] = 1
        bomb_flame[m, n-f[2]:n+f[3]+1] = 1

        return bomb_flame

    def update(self, gamma, batch_size):
        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        statesl, statesa, actions, rewards, next_statesl, next_statesa, done = self.buffer.sample(batch_size)
        #print(rewards)
        #print(actions)
        action_index = actions.squeeze(-2)
        #print(action_index)
        curr_Q_batch = self.eval_net(statesl, statesa)#[:,0]
        curr_Q = curr_Q_batch.gather(1, action_index.type(torch.int64)).squeeze(-1)
        
        next_batch = self.target_net(next_statesl, next_statesa)#[:,0]
        next_Q = torch.max(next_batch,1)[0]
        #计算reward
        computed_reward = []
        for l, a, action, sl, sa, re in zip(next_statesl, next_statesa, actions, statesl, statesa, rewards):
            computed_reward.append(self.reward(l, a, action, sl, sa, re))


        #这是得到的reward
        computed_reward = torch.tensor(computed_reward)
        
        rewards_batch = computed_reward
        #print(rewards_batch)
        
        
        # expected_Q = rewards + self.gamma * torch.max(next_Q, 1)
        expected_Q = (gamma * next_Q + rewards_batch) * ~done + done * rewards_batch
        # max_q_prime = next_Q.max(1)[0].unsqueeze(1)
        # expected_Q = done * (rewards + gamma * max_q_prime) + (1 - done) * 1 / (1 - gamma) * rewards
        # expected_Q = done * (rewards + gamma * max_q_prime) + 1 / (1 - gamma) * rewards
        loss = self.MSE_loss(curr_Q, expected_Q[0]) # TODO: try Huber Loss later too
        # print('loss:', loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def epsdecay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_eps else self.epsilon

    # def lrdecay(self,epi):
    #     epi = self.epi
    #     self.lr = self.lr * self.lr_decay ** (epi) if s

    def save_model(self):
        torch.save({'dqn2Net': self.eval_net.state_dict(),'optimizer2_state_dict': self.optim.state_dict()}, 'model_dqn2.pt')
    
    def load_model(self):
        if os.path.exists('model_dqn2.pt'):
            state_dict = torch.load('model_dqn2.pt')
            self.eval_net.load_state_dict(state_dict['dqn2Net'])
            self.optim.load_state_dict(state_dict['optimizer2_state_dict'])
            self.target_net.load_state_dict(self.eval_net.state_dict())

class Net2(nn.Module):
    def __init__(self,env):
        super(Net2,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(14,32,2,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(75, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, lx, ax):
        lx = torch.FloatTensor(lx)
        ax = torch.FloatTensor(ax)
        #lx = lx.unsqueeze(3)
        #print(lx[0])
        #x = torch.unsqueeze(x, dim=0).float()
        lx = self.features(lx)
        lx = lx.view(lx.size(0), -1)
        outx = torch.cat((lx, ax), 1)
        out = self.fc(outx)
        return out

    def forward1(self, lx, ax):
        lx = torch.FloatTensor(lx)
        ax = torch.FloatTensor(ax)
        lx = lx.unsqueeze(0)
        lx = self.features(lx)
        lx = lx.view(lx.size(0), -1)
        ax = ax.unsqueeze(0)
        outx = torch.cat((lx, ax), 1)
        out = self.fc(outx)
        return out