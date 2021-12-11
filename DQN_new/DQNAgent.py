from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents.random_agent import RandomAgent
from pommerman import characters

from gym.spaces import Discrete

from DQN_new import constants
from replay_memory import replay_Memory
import numpy as np


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = RandomAgent()

        self.training_model = self.new_model()
        self.trained_model = self.new_model()
        # self.trained_model.set_weights(self.training_model.get_weights())

        self.epsilon = constants.epsilon
        self.min_epsilon = constants.MIN_EPSILON
        self.eps_decay = constants.EPSILON_DECAY
        self.buffer = replay_Memory(constants.MAX_BUFFER_SIZE)
        self.target_update_counter = 0

    def new_model(self):

        model = Sequential()
        input_shape = (constants.MINIBATCH_SIZE, 12, 8, 8)
        model.add(Conv2D(256, 3, input_shape=input_shape[1:], activation="relu"))
        # print(model.output_shape)
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, 2, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(6, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self, done):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        # 取样
        mini_batch = self.buffer.sample(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.training_model.predict(current_states)

        # 在样品中取 next_state, 从网络中获取Q值
        next_state = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.trained_model.predict(next_state)

        # X为state，Y为所预测的action
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):

            # 如果 done， 则不会有future_Q
            if not done:
                # 更新Q值
                max_future_q = np.max(future_qs_list[index])
                next_q = reward + constants.DISCOUNT * max_future_q
            else:
                next_q = reward

            # 在给定的states下更新Q值
            current_qs = current_qs_list[index]
            current_qs[action] = next_q

            # 添加训练数据
            X.append(current_state)
            y.append(current_qs)

        # 开始训练
        self.training_model.fit(np.array(X), np.array(y), batch_size=constants.MINIBATCH_SIZE, verbose=0, shuffle=False)

        # 更新网络更新计数器
        if done:
            self.target_update_counter += 1

        # 网络更新计数器达到上限，更新网络
        if self.target_update_counter > constants.UPDATE_TARGET_EVERY:
            self.trained_model.set_weights(self.training_model.get_weights())
            self.target_update_counter = 0

    def get_q_value(self, state):

        return self.training_model.predict_on_batch(np.array(state).reshape(-1, 12, 8, 8))

        # epsilon衰减

    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon
