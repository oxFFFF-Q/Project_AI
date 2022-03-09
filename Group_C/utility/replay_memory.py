import random
import collections
import numpy as np
from Group_C.utility import constants


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.n_step = constants.n_step
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.n_step_buffer = collections.deque(maxlen=self.n_step)
        self.buffer_episode = collections.deque()
        self.buffer_processing = collections.deque()
        self.buffer_action = collections.deque([0, 0, 0, 0], maxlen=4)
        self.buffer_td = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.alpha = 0.6
        self.gamma = 0.9  # affinity for long term reward

    def append(self, transition):
        self.buffer.append(transition)

    def append_n_step(self, state, action, reward, next_state, done):
        # multi steps DQN
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        l_state, l_action = self.n_step_buffer[0][:2]
        transition_ = (l_state, l_action, l_reward, l_next_state, l_done)
        self.buffer.append(transition_)

    def append_action(self, action):
        self.buffer_action.append(action)

    def append_augmentation(self, transition):
        self.buffer_episode.append(transition)

    def append_processing(self, transition):
        self.buffer_processing.append(transition)

    def append_pri(self, state, action, reward, next_state, done, td_error):
        # pri DQN
        transition = (state, action, reward, next_state, done)
        self.append_td(td_error)
        self.buffer.append(transition)

    def append_td(self, td_error):
        self.buffer_td.append(td_error)

    def clear(self):
        self.buffer_episode.clear()
        self.buffer_processing.clear()

    def merge(self):
        for element in self.buffer_processing:
            self.buffer.append(element)

    def merge_negative(self):
        for element in self.buffer_processing:
            if element[2] < 0:
                self.buffer.append(element)

    def sample(self, batch):
        mini_batch = random.sample(self.buffer, batch)
        return mini_batch

    def sample_element(self, batch):
        mini_batch = random.sample(self.buffer, batch)
        current_state, action, reward, new_states, done = [], [], [], [], []

        for transition in mini_batch:
            curr_state, act, r, new_state, d = transition
            current_state.append(curr_state)
            action.append(act)
            reward.append(r)
            new_states.append(new_state)
            done.append(d)

        return np.array(current_state), action, reward, np.array(new_states), done

    def sample_element_pri(self, batch_size):
        # Prioritized DQN
        # 根据td_error排序，求出索引index, 从小到大
        index = np.argsort(np.array(self.buffer_td).flatten()).tolist()
        # buffer 按index排序
        buffer_sort = self.buffer
        if len(index) != 0 and len(buffer_sort) != 0:
            for i in range(len(self.buffer)):
                buffer_sort[i] = self.buffer[index[i]]
        prioritization = int(batch_size * self.alpha)  # self.alpha = 0.6
        batch_prioritized = []
        for i in range(prioritization):
            # 反向添加，从大到小
            batch_prioritized.append(buffer_sort[-i - 1])
        mini_batch = random.sample(self.buffer, batch_size - prioritization)
        td = self.buffer_td
        # 最训练使用数据batch= batch_prioritized(按td_error从大到小)+mini_batch(随机抽取)
        batch = batch_prioritized + mini_batch
        current_state, action, reward, new_states, done, td_error = [], [], [], [], [], []

        for transition in batch:
            curr_state, act, r, new_state, d = transition
            current_state.append(curr_state)
            action.append([act])
            reward.append([r])
            new_states.append(new_state)
            done.append([d])

        return np.array(current_state), action, reward, np.array(new_states), done

    def size(self):
        return len(self.buffer)
