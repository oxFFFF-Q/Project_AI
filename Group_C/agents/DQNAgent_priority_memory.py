from keras.layers import Dense, Flatten, Conv2D
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents import RandomAgent
from pommerman import characters
from gym.spaces import Discrete
from Group_C.utility.replay_memory import replay_Memory
from Group_C.utility import constants
import numpy as np
import tensorflow as tf
import copy


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = RandomAgent()

        self.training_model = self.new_model()
        self.trained_model = self.new_model()
        self.trained_model.set_weights(self.training_model.get_weights())
        # self.load_weights()

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
        return model

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        current_states, action, reward, new_states, done = self.buffer.sample_element_pri(constants.MINIBATCH_SIZE)

        # Take the current_states in the sample, get the Q value from the model
        current_states_q = self.training_model.predict(current_states)

        # Take next_state in the sample, get the Q value from the old network
        new_states_q = self.trained_model.predict(new_states)

        # X is the state, Y is the predicted action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] is not True:
                # Update Q value
                new_state_q = reward[index] + constants.DISCOUNT * np.max(new_states_q[index])
            else:
                new_state_q = reward[index]

            # Update the Q value for the given states
            current_better_q = current_states_q[index]
            current_better_q[action[index]] = new_state_q

            # Add training data
            states.append(current_states[index])
            actions.append(current_better_q)

        # Start training
        self.training_model.fit(np.array(states), np.array(actions), batch_size=constants.MINIBATCH_SIZE, verbose=0,
                                shuffle=False)

        # Update network update counters
        if done:
            self.update_counter += 1

        # Network update counter reaches upper limit, update network
        if self.update_counter > constants.UPDATE_EVERY:
            self.trained_model.set_weights(self.training_model.get_weights())
            self.update_counter = 0

    def calculate_td_error(self, state, action, reward, new_state, done):
        state_ = tf.reshape(state, (-1, 18, 11, 11))
        new_state_ = tf.reshape(new_state, (-1, 18, 11, 11))
        q_values = self.training_model.predict(state_)[0]
        q_value = q_values[action]
        if not done:
            target = reward + constants.DISCOUNT * \
                     self.trained_model.predict(new_state_)[0, np.argmax(self.training_model.predict(new_state_))]
        else:
            target = reward

        td_error = target - q_value
        return td_error

    def action_choose(self, state):

        state_reshape = tf.reshape(state, (-1, 18, 11, 11))
        q_table = self.training_model.predict(state_reshape)
        if np.random.random() <= 0.001:
            print(q_table)
        return q_table

    def save_buffer(self, state_feature, action, reward, next_state_feature, done):
        td_error = self.calculate_td_error(state_feature, action, reward, next_state_feature, done)
        self.buffer.append_pri(state_feature, action, reward, next_state_feature, done, td_error)

    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save_weights(self, numOfEpisode):

        # Archive parameters after training
        # save weight every "save_weight" episode, change it in constants.py
        if numOfEpisode % constants.save_weight == 0:
            self.training_model.save_weights(('./checkpoints/episode{:}/episode{:}'.format(numOfEpisode, numOfEpisode)))
            print("weights saved!")

    def load_weights(self, weight_name):
        self.training_model.load_weights('./checkpoints/{:}/{:}'.format(weight_name, weight_name))
        self.trained_model.load_weights('./checkpoints/{:}/{:}'.format(weight_name, weight_name))
        print("weights loaded!")

    def save_model(self, model_name):
        self.training_model.save("./{:}".format(model_name))
