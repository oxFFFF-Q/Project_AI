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
        model.add(Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
        model.add(Conv2D(256, 3, (1, 1), activation="relu", data_format="channels_first", padding="same"))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(6, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model.summary()
        return model

    def action_choose(self, state):

        state_reshape = tf.reshape(state, (-1, 18, 11, 11))
        q_table = self.training_model.predict(state_reshape)
        if np.random.random() <= 0.001:
            print(q_table)
        return q_table

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def save_buffer(self, state_feature, action, reward, next_state_feature, done):
        self.buffer.append([state_feature, action, reward, next_state_feature, done])

    def train(self):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        current_states, action, reward, new_states, done = self.buffer.sample_element(constants.MINIBATCH_SIZE)

        # 在样品中取 current_states, 从模型中获取Q值
        current_states_q = self.training_model.predict(current_states)

        # 在样品中取 next_state, 从旧网络中获取Q值
        new_states_q = self.trained_model.predict(new_states)

        # X为state，Y为所预测的action
        states = []
        actions = []

        for index in range(constants.MINIBATCH_SIZE):

            if done[index] is not True:
                # 更新Q值
                new_state_q = reward[index] + constants.DISCOUNT * np.max(new_states_q[index])
            else:
                new_state_q = reward[index]

            # 在给定的states下更新Q值
            current_better_q = current_states_q[index]
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

    # epsilon衰减
    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon

    def save_weights(self, numOfEpisode):
        # 完成训练后存档参数
        if numOfEpisode % 100 == 0:
            self.training_model.save_weights(('./checkpoints/FFA{:}/FFA{:}'.format(numOfEpisode, numOfEpisode)))
            print("weights saved!")

    def load_weights(self):
        self.training_model.load_weights('./checkpoints/FFA800/FFA800')
        self.trained_model.load_weights('./checkpoints/FFA800/FFA800')
        print("weights loaded!")

    def save_model(self):
        self.training_model.save("./Formal1")

    def data_processing(self, steps, reward, result, episode, copy_and_filter=False, convert_4_corner=False):
        # 将左上角地图转换到其他位置
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

        self.buffer.buffer_processing = copy.deepcopy(self.buffer.buffer_episode)
        if convert_4_corner is True:
            for state, action, reward, next_state, done in self.buffer.buffer_episode:
                # 旋转
                state_left_bottom, next_state_left_bottom, action_left_bottom = convert_left_bottom(state, next_state,
                                                                                                    action)
                state_right_bottom, next_state_right_bottom, action_right_bottom = convert_right_bottom(state,
                                                                                                        next_state,
                                                                                                        action)
                state_right_top, next_state_right_top, action_right_top = convert_right_top(state, next_state, action)
                self.buffer.append_processing(
                    [state_left_bottom, action_left_bottom, reward, next_state_left_bottom, done])
                self.buffer.append_processing(
                    [state_right_bottom, action_right_bottom, reward, next_state_right_bottom, done])
                self.buffer.append_processing([state_right_top, action_right_top, reward, next_state_right_top, done])

        if copy_and_filter is True:
            # 不考虑失败的游戏最后一步 -1的reward
            if result == 0:
                reward += 1

            average_reward = reward / steps
            if episode >= 0:
                # 若平均reward小于阈值，则正常存一份memory
                if average_reward <= constants.Threshold_max:
                    self.buffer.merge()
                else:
                    self.buffer.merge_negative()
                # 若reward小于阈值，则多拷贝一份memory
                if average_reward <= constants.Threshold_min:
                    self.buffer.merge()

            # 若充分训练，不放入memory
            # elif reward /steps >= constants.Threshold_max:
            #     pass
        else:
            self.buffer.merge()

        self.buffer.clear()
