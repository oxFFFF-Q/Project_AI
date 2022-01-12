from keras.layers import Dense, Flatten, Conv2D
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents.random_agent import RandomAgent
from pommerman.agents.simple_agent import SimpleAgent
from pommerman import characters

from gym.spaces import Discrete

from DQN_mulit_tensorflow_2 import constants
from replay_memory import replay_Memory
import numpy as np
import os
import tensorflow as tf


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = SimpleAgent()

        self.training_model = self.new_model()
        self.trained_model = self.new_model()

        self.trained_model.set_weights(self.training_model.get_weights())
        self.load_weights()

        self.epsilon = constants.epsilon
        self.min_epsilon = constants.MIN_EPSILON
        self.eps_decay = constants.EPSILON_DECAY
        self.buffer = replay_Memory(constants.MAX_BUFFER_SIZE)
        self.update_counter = 0

    def new_model(self):

        model = Sequential()
        input_shape = (constants.MINIBATCH_SIZE, 18, 11, 11,)
        model.add(Conv2D(256, 3, (1, 1), input_shape=input_shape[1:], activation="relu", data_format="channels_first",
                         padding="same"))
        # print(model.output_shape)
        model.add(Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
        # print(model.output_shape)
        model.add(Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
        # print(model.output_shape)

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(6, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model.summary()
        return model

    # def new_model(self):
    #
    #     model = Sequential()
    #     input_shape = (constants.MINIBATCH_SIZE, 18, 11, 11,)
    #     model.add(Conv2D(32, 3, (1, 1), input_shape=input_shape[1:], activation="relu", data_format="channels_first",
    #                      padding="same"))
    #     # print(model.output_shape)
    #     model.add(Conv2D(64, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
    #     # print(model.output_shape)
    #     model.add(Conv2D(64, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
    #     # print(model.output_shape)
    #
    #     model.add(Flatten())
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(6, activation='linear'))
    #     model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    #     model.summary()
    #     return model

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self):
        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_q_list = self.training_model.predict(current_states)

        # 在样品中取 next_state, 从旧网络中获取Q值
        future_q_list = self.trained_model.predict(new_states)

        # X为state，Y为所预测的action
        X = []
        Y = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] != True:
                # 更新Q值
                next_Q = reward[index] + constants.DISCOUNT * np.max(future_q_list[index])
            else:
                next_Q = reward[index]
            # 在给定的states下更新Q值
            current_qs = current_q_list[index]
            current_qs[action[index]] = next_Q

            # 添加训练数据
            X.append(current_states[index])
            # X.append(tf.reshape(current_state,(-1,14,11,11)))
            Y.append(current_qs)

            # 开始训练
        # 使用专用的数据api，但更慢.
        # X = tf.reshape(X, (-1, 12, 8, 8))
        # train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        # self.training_model.fit(train_dataset, verbose=0, shuffle=False)

        self.training_model.fit(np.array(X), np.array(Y), batch_size=constants.MINIBATCH_SIZE, verbose=0,
                                shuffle=False)

        # 更新网络更新计数器
        if done:
            self.update_counter += 1

        # 网络更新计数器达到上限，更新网络
        if self.update_counter > constants.UPDATE_EVERY:
            self.trained_model.set_weights(self.training_model.get_weights())
            self.update_counter = 0

    def action_choose(self, state):
        state_reshape = tf.reshape(state, (-1, 18, 11, 11))
        q_table = self.training_model.predict_on_batch(state_reshape)
        #print("q_table: ", q_table)
        return q_table
        #return self.training_model.predict_on_batch(state_reshape)

        # epsilon衰减

    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save_weights(self, numOfEpisode):

        # 完成训练后存档参数
        if numOfEpisode % 500 == 0:
            self.training_model.save_weights(('./checkpoints/FFA{:}/FFA{:}'.format(numOfEpisode, numOfEpisode)))
            #self.training_model.save_weights(('./checkpoints/FFA-test-1/FFA-test-1'.format(numOfEpisode, numOfEpisode)))

    def load_weights(self):
        self.training_model.load_weights('./checkpoints/FFA1500/FFA1500')
        self.trained_model.load_weights('./checkpoints/FFA1500/FFA1500')
        print("weights loaded!")
