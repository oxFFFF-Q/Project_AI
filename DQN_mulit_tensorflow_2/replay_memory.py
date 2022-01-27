import random
import collections
import numpy as np
import constants


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.n_step = constants.n_step
        self.epi_buffer = []
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.buffer_n_step = collections.deque(maxlen=self.n_step)
        self.buffer_action = collections.deque([0, 0, 0, 0], maxlen=4)
        self.buffer_td = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.alpha = 0.6

    def append_epi(self, transition):
        self.epi_buffer.append(transition)

    def append(self, transition):
        self.buffer.append(transition)

    def append_n_step(self, transition):
        self.buffer_n_step.append(transition)

    def append_action(self, action):
        self.buffer_action.append(action)

    def append_td(self, td_error):
        self.buffer_td.append(td_error)

    def reset_epi_buffer(self):
        self.epi_buffer = []

    def extract_buffer(self):
        index = int(len(self.epi_buffer)*0.3)
        self.buffer += self.epi_buffer[-index:]



    def sample(self, batch):
        mini_batch = random.sample(self.buffer, batch)

        return mini_batch

    def sample_element(self, batch):
        mini_batch = random.sample(self.buffer, batch)
        current_state, action, reward, new_states, done = [], [], [], [], []

        for transition in mini_batch:
            curr_state, act, r, new_state, d = transition
            current_state.append(curr_state)
            action.append([act])
            reward.append([r])
            new_states.append(new_state)
            done.append([d])

        return np.array(current_state), action, reward, np.array(new_states), done

    def sample_element_pre(self, batch_size):
        # Prioritized DQN
        # 根据td_error排序，求出索引index
        index = np.argsort(np.array(self.buffer_td).flatten()).tolist()
        # buffer 按index排序
        buffer_sort = self.buffer
        for i in range(len(self.buffer)):
            buffer_sort[i] = self.buffer[index[i]]
        prioritization = int(batch_size * self.alpha)
        batch_prioritized = []
        for i in range(prioritization):
            batch_prioritized.append(buffer_sort[-i-1])
        mini_batch = random.sample(self.buffer, batch_size-prioritization)
        td = self.buffer_td
        # 最训练使用数据batch= batch_prioritized(按td_error从大到小)+mini_batch(随机抽取)
        batch = batch_prioritized+mini_batch
        current_state, action, reward, new_states, done, td_error = [], [], [], [], [], []

        for transition in batch:
            curr_state, act, r, new_state, d= transition
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

    def size_n_step(self):
        return len(self.buffer_n_step)

    def save(self):
        current_states = [transition[0] for transition in self.buffer_pre]
        actions = [transition[1] for transition in self.buffer_pre]
        np.save("pre_trained/FFA-Dataset-states", current_states, allow_pickle=True)
        np.save("pre_trained/FFA-Dataset-actions", actions, allow_pickle=True)
