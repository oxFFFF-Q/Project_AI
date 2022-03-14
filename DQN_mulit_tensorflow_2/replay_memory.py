import random
import collections
import numpy as np
import constants


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.buffer_episode = collections.deque()
        self.buffer_processing = collections.deque()
        self.buffer_action = collections.deque([0, 0, 0, 0], maxlen=4)
        self.buffer_td = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.alpha = 0.6
        self.n_step = 4
        self.gamma = 0.9

    def append(self, transition):
        self.buffer.append(transition)

    def append_action(self, action):
        self.buffer_action.append(action)

    def append_augmentation(self, transition):
        self.buffer_episode.append(transition)

    def append_processing(self, transition):
        self.buffer_processing.append(transition)

    def append_pri(self, state, action, reward, next_state, done, td_error):
        # pri DQN
        transition = [state, action, reward, next_state, done]
        self.append_td(td_error)
        self.buffer.append(transition)
        return True

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

    def sample_all(self):
        # for pre-train
        current_state = [transition[0] for transition in self.buffer]
        action = [transition[0] for transition in self.buffer]

        return current_state, action

    def size(self):
        return len(self.buffer)
