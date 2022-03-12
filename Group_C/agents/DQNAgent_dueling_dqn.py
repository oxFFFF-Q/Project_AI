from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents import RandomAgent
from pommerman import characters
from gym.spaces import Discrete
from Group_C.utility.replay_memory import replay_Memory
from Group_C.utility import constants
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class Dueling_Model(tf.keras.Model):

    def __init__(self, ):
        super(Dueling_Model, self).__init__()

        self.c1 = keras.layers.Conv2D(256, 3, (1, 1), input_shape=(constants.MINIBATCH_SIZE, 18, 11, 11,)[1:],
                                      activation="relu", data_format="channels_first",
                                      padding="same")
        self.c2 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")

        self.c3 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")
        self.flatten = keras.layers.Flatten()
        self.l1 = keras.layers.Dense(128, activation="relu")

        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(6, activation=None)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.l1(x)

        V = self.V(x)
        A = self.A(x)
        mean = tf.math.reduce_mean(A, axis=1, keepdims=True)

        output = V + tf.subtract(A, mean)
        return output

    def advantage(self, state):
        x = self.c1(state)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.l1(x)
        A = self.A(x)
        return A


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = RandomAgent()

        self.training_model = Dueling_Model()
        self.trained_model = Dueling_Model()

        self.trained_model.set_weights(self.training_model.get_weights())
        # self.load_weights()

        self.epsilon = constants.epsilon
        self.min_epsilon = constants.MIN_EPSILON
        self.eps_decay = constants.EPSILON_DECAY
        self.buffer = replay_Memory(constants.MAX_BUFFER_SIZE)
        self.update_counter = 0

        self.training_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        self.trained_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_states_q = self.training_model.call(current_states)
        # double_new_qs = self.training_model.call(new_states)

        # 在样品中取 next_state, 从旧网络中获取Q值
        # new_states_q = tf.math.reduce_max(self.trained_model.call(new_states), axis=1, keepdims=True)
        new_states_q = self.trained_model.call(new_states)

        # X为state，Y为所预测的action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] is not True:
                # 更新Q值
                new_state_q = reward[index] + constants.DISCOUNT * np.max(new_states_q[index])
            else:
                new_state_q = reward[index]

            # estimate q-values based on current state
            q_values = current_states_q[index]
            # 在给定的states下更新Q值
            current_better_q = np.array(q_values)
            current_better_q[action[index]] = new_state_q

            # 添加训练数据
            states.append(current_states[index])
            actions.append(current_better_q)

        # 开始训练
        self.training_model.fit(np.array(states), np.array(actions), batch_size=constants.MINIBATCH_SIZE, verbose=0,
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
        q_table = self.training_model.advantage(state_reshape)
        if np.random.random() <= 0.001:
            print(q_table)
        return q_table

    # epsilon衰减
    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save_buffer(self, state_feature, action, reward, next_state_feature, done):
        self.buffer.append([state_feature, action, reward, next_state_feature, done])

    def save_weights(self, numOfEpisode):

        # 完成训练后存档参数
        if numOfEpisode % 100 == 0:
            self.training_model.save_weights(('./checkpoints/FFA{:}/FFA{:}'.format(numOfEpisode, numOfEpisode)))
            print("weights saved!")

    def load_weights(self):
        self.training_model.load_weights('./checkpoints/FFA400/FFA400')
        self.trained_model.load_weights('./checkpoints/FFA400/FFA400')
        print("weights loaded!")

    def save_model(self):
        self.training_model.save("./agent1nofilter")
