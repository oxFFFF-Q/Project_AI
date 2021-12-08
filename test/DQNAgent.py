from pommerman.agents.simple_agent import SimpleAgent #random_agent
import torch
import torch.nn as nn
import torch.optim as optim
from pommerman import characters
import torch.nn.functional as F

import random
import gym
import numpy as np
from utils import featurize

from pommerman.agents import BaseAgent
from replay_buffer import ReplayBuffer, ReplayBuffer2


class DQNAgent(BaseAgent):
    """
    DQN from scratch
    """
    def __init__(self, env, args, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.obs_n = env.observation_space.shape[0]  # output_dim
        self.action_n = env.action_space.n  # input_dim
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
        self.eval_net, self.target_net = Net(), Net()
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
        lx = obs['local']
        ax = obs['additional']
        action = self.eval_net.forward1(lx,ax)[0]
        result = (torch.max(action, 0)[1]).numpy()
        return result

    def reward(self, featurel, featurea, action, sl, sa, epistep):
        rigid = featurel[0]
        wood = featurel[1]
        bomb = featurel[2]
        agents = featurel[4]
        power_up = featurel[3]
        position0 = int(featurea[0].item())
        position1 = int(featurea[1].item())
        ammo = int(featurea[2].item())
        blast_strength = int(featurea[3].item())
        can_kick = int(featurea[4].item())
        teammate = int(featurea[5].item())
        enemies = int(featurea[6].item())
        reward = 0
        sagents = sl[4]
        sbombs = sl[2]
        es = epistep.tolist()
        es[1] += 2
        
        #道具
        sammo = int(sa[2].item())
        if ammo > sammo and ammo > 1:
            reward += 0.3
        
        #print(position0, position1)
        #print(int(bomb[int(position0),  int(position1)]))
        if int(action[0].item()) == 5:
            m = position0
            n = position1
            l = blast_strength
            f = [0,7,0,7]       # Scope of flame: up down left right
            
            bomb_withflame = np.zeros_like(agents)
            bomb_withflame[m, n] = 1
            # 假设火焰无限大,判断实体墙是否阻断火焰
            flame_up = bomb_withflame
            flame_down = bomb_withflame
            flame_left = bomb_withflame
            flame_right = bomb_withflame
            flame_up[0:m,n] = 1
            fr1 = flame_up*rigid.numpy()
            flame_down[m:,n] = 1
            flame_left[m,0:n] = 1
            flame_right[m,n:] = 1
            fr1 = np.argwhere(fr1==1)
            if fr1.size != 0:
                f[0] = np.max(fr1[:,0])
                print(f[0])
            if np.argwhere(flame_down*rigid.numpy()==1).size != 0:
                f[1] = np.min(np.argwhere(flame_down*rigid.numpy()==1)[:,0])
                print(f[1])
            if np.argwhere(flame_left*rigid.numpy()==1).size != 0:
                f[2] = np.max(np.argwhere(flame_left*rigid.numpy()==1)[0,:])
                print(f[2])
            if np.argwhere(flame_right*rigid.numpy()==1).size != 0:
                f[3] = np.min(np.argwhere(flame_right*rigid.numpy()==1)[0,:])
                print(f[3])
            # 判断火焰是否出界,顺序：上下左右
            if m-l < 0 and f[0] == 0:
                bomb_withflame[0:m,n] = 1
            elif f[0] == 0:
                bomb_withflame[m-l:m,n] = 1
            else:
                bomb_withflame[f[0]+1:m, n] = 1
            if m+l > 7:
                bomb_withflame[m:,n] = 1
            else:
                bomb_withflame[m:f[1]-1, n] = 1
            if n-l < 0:
                bomb_withflame[m,0:n] = 1
            else:
                bomb_withflame[m,f[2]+1:n] = 1
            if m+f[3] > 7:
                bomb_withflame[m,n:] = 1
            else:
                bomb_withflame[m,n:f[3]-1] = 1
            num_wood = np.count_nonzero(wood.numpy()*bomb_withflame == 1)
            print("map")
            print(bomb_withflame)
            print(rigid)
            reward += num_wood
        
        return reward

    def update(self, gamma, batch_size,episode, step):
        #每走十步学习一次
        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        statesl, statesa, actions, rewards, next_statesl, next_statesa, done, epistep = self.buffer.sample(batch_size)
        #print(epistep)
        
        #计算reward
        computed_reward = []
        for l, a, action, sl, sa, es in zip(next_statesl, next_statesa, actions, statesl, statesa, epistep):
            computed_reward.append(self.reward(l, a, action[0], sl, sa, es))
        
        computed_reward = torch.tensor(computed_reward)
        action_index = actions.squeeze(-2)[:,0].unsqueeze(1)
        curr_Q_batch = self.eval_net(statesl,statesa)#[:,0]
        #print(curr_Q_batch)
        curr_Q = curr_Q_batch.gather(1, action_index.type(torch.int64)).squeeze(-1)
        
        next_batch = self.target_net(next_statesl, next_statesa)#[:,0]
        next_Q = torch.max(next_batch,1)[0]

        rewards_batch = rewards.squeeze(-2)[:,0]
        rewards_batch = rewards_batch + computed_reward
        #print(rewards_batch)
        #print(rewards_batch)
        # expected_Q = rewards + self.gamma * torch.max(next_Q, 1)
        #需要把done计算进去
        expected_Q = (gamma * next_Q + rewards_batch) * ~done + done * rewards_batch

        # max_q_prime = next_Q.max(1)[0].unsqueeze(1)
        # expected_Q = done * (rewards + gamma * max_q_prime) + (1 - done) * 1 / (1 - gamma) * rewards
        # expected_Q = done * (rewards + gamma * max_q_prime) + 1 / (1 - gamma) * rewards
        loss = self.MSE_loss(curr_Q, expected_Q[0]) # TODO: try Huber Loss later too

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def epsdecay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_eps else self.epsilon

    def compute_reward(self, local, additional, epistep):
        m = self.buffer.get(tuple(epistep.tolist()))
        print("element in memory:")
        print(m)
        return 0


class Net1(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        """
        self.conv1=nn.Conv2d(199,16,2,stride=1,padding=1)
        self.conv2=nn.Conv2d(16,32,3,stride=1,padding=1)
        """
        self.fc1 = nn.Linear(199,128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128,6)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        #x = torch.unsqueeze(x, dim=0).float()
        #x=self.conv1(x)
        #x=self.conv2(x)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x)
        return out
    
    

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(9,32,2,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(71, 32),
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
        #lx = lx.unsqueeze(3)
        #x = torch.unsqueeze(x, dim=0).float()
        lx = self.features(lx)
        lx = lx.view(lx.size(0), -1)
        ax = ax.unsqueeze(0)
        outx = torch.cat((lx, ax), 1)
        out = self.fc(outx)
        return out
        
