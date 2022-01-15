import numpy as np
import random
from collections import deque
import torch


# # 可以使用index or name进行访问
# Experience = collections.namedtuple('Experience',
#                                     field_names=['state', 'action', 'reward',
#                                                  'done', 'new_state'])


# class ReplayBuffer(object):
#     """
#     Implementing DQN's Replay Buffer from scratch
#     """
#
#     def __init__(self, capacity, batch_size):
#         self.buffer = deque(maxlen=capacity)
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def append(self, experience):
#         self.buffer.append(experience)
#         # 数据覆盖
#         if len(self.buffer) > self.buffer.maxlen:
#             self.buffer.popleft()                 # 删去从最左边的数据
#
#     def sample(self, batch_size):
#         state_batch = []
#         action_batch = []
#         reward_batch = []
#         next_state_batch = []
#         done_batch = []
#
#         batch = random.sample(self.buffer, batch_size)  # 从序列seq中选择n个随机且独立的元素
#
#         for experience in batch:
#             state, action, reward, next_state, done = experience
#             state_batch.append(state)
#             action_batch.append(action)
#             reward_batch.append(reward)
#             next_state_batch.append(next_state)
#             done_batch.append(done)
#
#         return (state_batch, action_batch, reward_batch, done_batch, next_state_batch)
#
#     def sample2(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return np.concatenate(state), action, reward, np.concatenate(next_state), done


class ReplayBuffer2():

    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)   # 从两端append 的数据,  默认从右边加入
        self.batch_size = batch_size

    def append(self, transition):
        self.buffer.append(transition)
        # 数据覆盖
        if len(self.buffer) > self.buffer.maxlen:
            self.buffer.popleft()  # 删去从最左边的数据

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)    # 从序列中选择n个随机且独立的元素
        sl_lst, sa_lst, a_lst, r_lst, sl_prime_list, sa_prime_list, done_mask_list = deque(), deque(), deque(), deque(), deque(), deque(), deque()
        # sl_lst, sa_lst, a_lst, r_lst, sl_prime_list, sa_prime_list, done_mask_list = [], [], [], [], [], [], []
        for transition in mini_batch:  # transition: tuple
            s, a, r, s_prime, done_mask = transition
            sl_lst.append(s['local'])
            sa_lst.append(s['additional'])
            a_lst.append([a])
            r_lst.append([r])
            sl_prime_list.append(s_prime['local'])
            sa_prime_list.append(s_prime['additional'])
            done_mask_list.append([done_mask])


        return (torch.tensor(sl_lst, dtype=torch.float),
                torch.tensor(sa_lst, dtype=torch.float),
                torch.tensor(a_lst), torch.tensor(r_lst),
                torch.tensor(sl_prime_list, dtype=torch.float),
                torch.tensor(sa_prime_list, dtype=torch.float),
                torch.tensor(done_mask_list))

    def size(self):
        return len(self.buffer)