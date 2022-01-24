import random
import collections
import numpy as np
import heapq
import pickle


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.buffer_action = collections.deque([0, 0, 0, 0], maxlen=4)
        self.buffer_td = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.alpha = 0.6

    def append(self, transition):
        self.buffer.append(transition)

    def append_action(self, action):
        self.buffer_action.append(action)

    def append_td(self, td_error):
        self.buffer_td.append(td_error)

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
        buffer = self.buffer
        index = np.argsort(np.array(self.buffer_td).flatten()).tolist()
        buffer_sort = buffer
        for i in range(len(buffer)):
            buffer_sort[i] = buffer[index[i]]
        prioritization = int(batch_size * self.alpha)
        batch_prioritized = []
        for i in range(prioritization):
            batch_prioritized.append(buffer_sort[-i-1])
        mini_batch = random.sample(self.buffer, batch_size-prioritization)
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

    def save(self):
        current_states = [transition[0] for transition in self.buffer_pre]
        actions = [transition[1] for transition in self.buffer_pre]
        np.save("pre_trained/FFA-Dataset-states", current_states, allow_pickle=True)
        np.save("pre_trained/FFA-Dataset-actions", actions, allow_pickle=True)
