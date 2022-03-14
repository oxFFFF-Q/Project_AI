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

        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # Take the current_states in the sample, get the Q value from the model
        current_states_q = self.training_model.predict(current_states)
        double_new_states_q = self.training_model.predict(new_states)
        # Take next_state in the sample, get the Q value from the old network
        new_states_q = self.trained_model.predict(new_states)

        # X is the state, Y is the predicted action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] is not True:
                # Update Q value, Double DQN
                target = reward[index] + constants.DISCOUNT * new_states_q[index][np.argmax(double_new_states_q[index])]
            else:
                target = reward[index]

            # estimate q-values based on current state
            q_values = current_states_q[index]

            # Update the Q value for the given states
            current_better_q = np.array(q_values)
            current_better_q[action[index]] = target

            # Add training data
            states.append(current_states[index])
            actions.append(current_better_q)

        # Use dedicated data api, but slower.
        # states = tf.reshape(states, (-1, 12, 8, 8))
        # train_dataset = tf.data.Dataset.from_tensor_slices((states, actions))
        # self.training_model.fit(train_dataset, verbose=0, shuffle=False)

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

    def save_buffer(self, state_feature, action, reward, next_state_feature, done, Data_processing=False):
        """if you want rotate observation into other 3 corner, set Data_processing to True"""
        if Data_processing:
            self.data_processing(state_feature, action, reward, next_state_feature, done)
        else:
            self.buffer.append([state_feature, action, reward, next_state_feature, done])

    def action_choose(self, state):

        state_reshape = tf.reshape(state, (-1, 18, 11, 11))
        q_table = self.training_model.predict(state_reshape)
        if np.random.random() <= 0.001:
            print(q_table)
        return q_table

    # epsilon decay
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

    def data_processing(self, state_feature, action, reward, next_state_feature, done):
        # Convert the top left map to another location
        def convert_left_bottom(state_feature, next_state_feature, action):
            state_feature_left_bottom = []
            for board in state_feature:
                state = np.rot90(board, k=1)
                state_feature_left_bottom.append(state)
            next_state_feature_left_bottom = []
            for board in next_state_feature:
                state = np.rot90(board, k=1)
                next_state_feature_left_bottom.append(state)
            if action == 1:
                action = 3
            elif action == 2:
                action = 4
            elif action == 3:
                action = 2
            elif action == 4:
                action = 1
            return np.array(state_feature_left_bottom), np.array(next_state_feature_left_bottom), action

        def convert_right_bottom(state_feature, next_state_feature, action):
            state_feature_left_bottom = []
            for board in state_feature:
                state = np.rot90(board, k=2)
                state_feature_left_bottom.append(state)
            next_state_feature_left_bottom = []
            for board in next_state_feature:
                state = np.rot90(board, k=2)
                next_state_feature_left_bottom.append(state)
            if action == 1:
                action = 2
            elif action == 2:
                action = 1
            elif action == 3:
                action = 4
            elif action == 4:
                action = 3
            return np.array(state_feature_left_bottom), np.array(next_state_feature_left_bottom), action

        def convert_right_top(state_feature, next_state_feature, action):
            state_feature_left_bottom = []
            for board in state_feature:
                state = np.rot90(board, k=3)
                state_feature_left_bottom.append(state)
            next_state_feature_left_bottom = []
            for board in next_state_feature:
                state = np.rot90(board, k=3)
                next_state_feature_left_bottom.append(state)
            if action == 1:
                action = 4
            elif action == 2:
                action = 3
            elif action == 3:
                action = 1
            elif action == 4:
                action = 2
            return np.array(state_feature_left_bottom), np.array(next_state_feature_left_bottom), action

            # Rotate

        state_left_bottom, next_state_left_bottom, action_left_bottom = convert_left_bottom(state_feature,
                                                                                            next_state_feature,
                                                                                            action)
        state_right_bottom, next_state_right_bottom, action_right_bottom = convert_right_bottom(state_feature,
                                                                                                next_state_feature,
                                                                                                action)
        state_right_top, next_state_right_top, action_right_top = convert_right_top(state_feature, next_state_feature,
                                                                                    action)
        self.buffer.append([state_feature, action, reward, next_state_feature, done])
        self.buffer.append([state_left_bottom, action_left_bottom, reward, next_state_left_bottom, done])
        self.buffer.append([state_right_bottom, action_right_bottom, reward, next_state_right_bottom, done])
        self.buffer.append([state_right_top, action_right_top, reward, next_state_right_top, done])