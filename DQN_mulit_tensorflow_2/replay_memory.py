import random
import collections
import numpy as np
import pickle


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)
        self.buffer_action = collections.deque([0, 0, 0, 0], maxlen=4)
        self.buffer_pre = []

    def append(self, transition):
        self.buffer.append(transition)

    def append_action(self, action):
        self.buffer_action.append(action)

    def append_pre(self, transition):
        self.buffer_pre.append(transition)

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
