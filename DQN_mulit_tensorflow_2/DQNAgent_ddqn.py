import tensorflow.keras as keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from pommerman.agents import BaseAgent
from pommerman.agents.simple_agent import SimpleAgent
from pommerman import characters

from gym.spaces import Discrete

import constants
from replay_memory import replay_Memory
import numpy as np
import tensorflow as tf

class Dueling_Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Dueling_Model, self).__init__(**kwargs)
        
        self.c1 = keras.layers.Conv2D(256, 3, (1, 1), input_shape=(constants.MINIBATCH_SIZE, 18, 11, 11,)[1:], activation="relu", data_format="channels_first",
                         padding="same")
        self.c2 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")
    
        self.c3 = keras.layers.Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same")

        self.flatten = keras.layers.Flatten()
        self.l1 = keras.layers.Dense(128, activation="relu")
        self.l2 = keras.layers.Dense(64, activation='linear')
        
        self.V = keras.layers.Dense(1,activation=None)
        self.A = keras.layers.Dense(6,activation=None)
        

    def call(self, inputs):
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
        #self.load_weights()

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
        
        #dueling
        
        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_states_q = self.training_model(current_states)
        double_new_qs = self.training_model(new_states)
        
        # 在样品中取 next_state, 从旧网络中获取Q值
        new_states_q = self.trained_model(new_states)
        
        # X为state，Y为所预测的action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] != True:
                # 更新Q值
                #new_state_q = reward[index] + constants.DISCOUNT * np.max(new_states_q[index])
                double_new_q = reward[index] + constants.DISCOUNT * new_states_q[index][np.argmax(double_new_qs[index])]
            else:
                #new_state_q = reward[index]
                double_new_q = reward[index]
            # 在给定的states下更新Q值
            current_better_q = current_states_q[index]
            current_better_q = current_better_q.numpy()
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
        self.training_model.load_weights('./checkpoints/FFA2200/FFA2200')
        self.trained_model.load_weights('./checkpoints/FFA2200/FFA2200')
        print("weights loaded!")

    def save_model(self):
        self.training_model.save("./second_model")
