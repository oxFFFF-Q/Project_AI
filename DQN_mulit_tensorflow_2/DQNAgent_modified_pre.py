import pickle
import numpy as np
import os
import tensorflow as tf

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents.random_agent import RandomAgent
from pommerman.agents.simple_agent import SimpleAgent
from pommerman import characters
from gym.spaces import Discrete
from sklearn.model_selection import train_test_split
from DQN_new import constants
from replay_memory import replay_Memory
from keras.utils import to_categorical


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = SimpleAgent()

        self.training_model = self.new_model()
        self.trained_model = self.new_model()

        self.epsilon = constants.epsilon
        self.min_epsilon = constants.MIN_EPSILON
        self.eps_decay = constants.EPSILON_DECAY
        self.buffer = replay_Memory(constants.MAX_BUFFER_SIZE)
        self.update_counter = 0

    def new_model(self):

        model = Sequential()
        input_shape = (constants.MINIBATCH_SIZE, 14, 11, 11,)
        model.add(Conv2D(64, 2, input_shape=input_shape[1:], activation="relu", data_format="channels_first"))
        print(model.output_shape)
        # model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_first"))
        model.add(Conv2D(64, 2, activation="relu", data_format="channels_first"))
        print(model.output_shape)
        # model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
        model.add(Conv2D(64, 2, activation="relu", data_format="channels_first"))
        print(model.output_shape)

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(6, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

        return model

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self, dataset_states, dataset_actions):
        current_states = dataset_states
        actions = dataset_actions

        # x_train, x_val, y_train, y_val = train_test_split(current_states, actions, test_size=0.2, shuffle=True)
        #
        # self.training_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, verbose=1, shuffle=True, batch_size=1024)

        self.training_model.fit(current_states, actions, epochs=15, verbose=1, shuffle=True, batch_size=1024)

    def save_weights(self):
        self.training_model.save_weights('./checkpoints/pre-train/pre-train')

    def save_data(self):
        self.buffer.save()
