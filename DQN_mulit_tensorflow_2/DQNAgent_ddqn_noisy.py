import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from gym.spaces import Discrete
from tensorflow.keras.optimizers import Adam

import constants
from pommerman import characters
from pommerman.agents import BaseAgent
from pommerman.agents.simple_agent import SimpleAgent
from replay_memory import replay_Memory


class Dueling_Model(tf.keras.Model):
    # Dueling DQN
    def __init__(self, **kwargs):
        super(Dueling_Model, self).__init__(**kwargs)

        self.c1 = keras.layers.Conv2D(256, 3, (1, 1), input_shape=(constants.MINIBATCH_SIZE, 18, 11, 11,)[1:],
                                      activation="relu", data_format="channels_first",
                                      padding="same")
        self.c2 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")

        self.c3 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")

        # Noisy network
        self.out = keras.layers.Flatten(self.c3)
        self.out1 = self.noisy_dense(self.out, size=512, name='noisy_fc1', activation_fn=tf.nn.relu)
        num_actions = 6
        self.out2 = self.noisy_dense(self.out1, size=num_actions, name='noisy_fc2')

        self.flatten = keras.layers.Flatten()
        self.l1 = keras.layers.Dense(128, activation="relu")
        self.l2 = keras.layers.Dense(64, activation='relu')

        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(6, activation=None)

    # for NoisyNet-DQN (using Factorised Gaussian noise)
    # modified from ```dense``` function
    def noisy_dense(self, x, size, name, bias=True, activation_fn=tf.identity):
        def sample_noise(shape):
            noise = tf.random_normal(shape)
            return noise

        # the function used in eq.7,8
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

        # Initializer of \mu and \sigma
        mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(x.get_shape().as_list()[1], 0.5),
                                                maxval=1 * 1 / np.power(x.get_shape().as_list()[1], 0.5))
        sigma_init = tf.constant_initializer(0.4 / np.power(x.get_shape().as_list()[1], 0.5))
        # Sample noise from gaussian
        p = sample_noise([x.get_shape().as_list()[1], 1])
        q = sample_noise([1, size])
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable(name + "/b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
            b = b_mu + tf.multiply(b_sigma, b_epsilon)
            return activation_fn(ret + b)
        else:
            return activation_fn(ret)

    def call(self, inputs: object) -> object:
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)

        V = self.V(x)
        # advantage value
        A = self.A(x)
        mean = tf.math.reduce_mean(A, axis=1, keepdims=True)
        # output
        # Duelling dqn
        output = V + (A - mean)
        return output

    def advantage(self, state):
        x = self.c1(state)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)
        A = self.A(x)
        return A


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.baseAgent = SimpleAgent()

        self.training_model = Dueling_Model()
        self.trained_model = Dueling_Model()

        self.trained_model.set_weights(self.training_model.get_weights())
        # self.load_weights()

        self.epsilon = constants.epsilon
        self.min_epsilon = constants.MIN_EPSILON
        self.eps_decay = constants.EPSILON_DECAY
        self.buffer = replay_Memory(constants.MAX_BUFFER_SIZE)
        self.update_counter = 0
        self.n_step = constants.n_step

        self.training_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        self.trained_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        # if self.buffer.size_n_step() < constants.n_step:
        #     return

        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_states_q = self.training_model.call(current_states)
        double_states_q = self.training_model.call(new_states)

        # 在样品中取 next_state, 从旧网络中获取Q值
        new_states_q = self.trained_model.call(new_states)

        # X为state，Y为所预测的action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] != True:
                # 更新Q值, Double DQN
                # new_state_q = reward[index] + constants.DISCOUNT * (np.max(new_states_q[index]) - current_states_q[index])
                # double dqn取最大值
                index_action = np.argmax(double_states_q[index])
                # 代入算出新的q值
                double_new_q = reward[index] + constants.DISCOUNT * new_states_q[index, index_action]
            else:
                # new_state_q = reward[index]
                double_new_q = reward[index]

            # 在给定的states下更新Q值
            current_better_q = current_states_q[index].numpy()
            current_better_q[action[index]] = double_new_q
            current_better_q = tf.convert_to_tensor(current_better_q)

            # 添加训练数据
            states.append(current_states[index])
            actions.append(current_better_q)

            # 开始训练
        # 使用专用的数据api，但更慢.
        # states = tf.reshape(states, (-1, 12, 8, 8))
        # train_dataset = tf.data.Dataset.from_tensor_slices((states, actions))
        # self.training_model.fit(train_dataset, verbose=0, shuffle=False)

        self.training_model.train_on_batch(np.array(states), np.array(actions))

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
        return q_table

    # epsilon衰减
    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save_weights(self, numOfEpisode):
        # 完成训练后存档参数
        if numOfEpisode % 200 == 0:
            self.training_model.save_weights(('./checkpoints/FFA{:}/FFA{:}'.format(numOfEpisode, numOfEpisode)))
            # self.training_model.save_weights(('./checkpoints/FFA-test-1/FFA-test-1'.format(numOfEpisode, numOfEpisode)))
            print("weights saved!")

    def load_weights(self):
        # 更改路径中的数字即可读取对应模型参数
        self.training_model.load_weights('./checkpoints/FFA400/FFA400')
        self.trained_model.load_weights('./checkpoints/FFA400/FFA400')
        print("weights loaded!")

    def save_model(self):
        self.training_model.save("./second_model")
