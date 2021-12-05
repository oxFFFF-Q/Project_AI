import random
import collections


class replay_Memory():
    def __init__(self, MAX_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=MAX_BUFFER_SIZE)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch):
        mini_batch = random.sample(self.buffer, batch)
        # curr_s, act, reward, next_s, done = [], [], [], [], []
        #
        # # transition: tuple
        # for transition in mini_batch:
        #     c_s, a, r, n_s, d = transition
        #     curr_s.append(c_s)
        #     act.append([a])
        #     reward.append([r])
        #     next_s.append(n_s)
        #     done.append([d])

        # 构建transition
        return mini_batch

    def size(self):
        return len(self.buffer)
