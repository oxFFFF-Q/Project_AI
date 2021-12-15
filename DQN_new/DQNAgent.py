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
import os
import tensorflow as tf


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
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, 2, activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(6, activation='softmax'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def act(self, obs, action_space):
        return self.baseAgent.act(obs, Discrete(6))

    def train(self, done, numOfEpisode):

        if self.buffer.size() < constants.MIN_REPLAY_MEMORY_SIZE:
            return

        # if numOfEpisode == 0:
        #     self.training_model.load_weights('./checkpoints/my_checkpoint')
        #     self.trained_model.load_weights('./checkpoints/my_checkpoint')

        if numOfEpisode % 999 == 0:
            checkpoint_path = "/checkpoints/training1/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # 检查点重用
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
        # 完成一次训练后存档参数
        if numOfEpisode == 4999:
            self.training_model.save_weights('/checkpoints/my_checkpoint')


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
        Y = []

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
            X.append(np.array(current_state))
            # X.append(tf.reshape(current_state,(-1,12,8,8)))
            Y.append(np.array(current_qs))

            # 开始训练
        # X = tf.reshape(X, (-1, 12, 8, 8))
        # train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        # self.training_model.fit(train_dataset, verbose=0, shuffle=False)

        self.training_model.fit(np.array(X), np.array(Y), epochs=4, batch_size=constants.MINIBATCH_SIZE, verbose=0,
                                shuffle=False)

        # 更新网络更新计数器
        if done:
            self.target_update_counter += 1

        # 网络更新计数器达到上限，更新网络
        if self.target_update_counter > constants.UPDATE_TARGET_EVERY:
            self.trained_model.set_weights(self.training_model.get_weights())
        self.target_update_counter = 0

    def get_q_value(self, state):
        state_reshape = np.array(state).reshape(-1, 12, 8, 8)
        return self.training_model.predict_on_batch(state_reshape)

        # epsilon衰减

    def reward(self, featurel, featurea, action, sl, sa, epistep):
        # set up reward
        r_wood = 0.1

        rigid = featurel[0].numpy()
        wood = featurel[1].numpy()
        bomb = featurel[2]
        agents = featurel[4]
        power_up = featurel[3]
        position0 = int(featurea[0].item())
        position1 = int(featurea[1].item())
        ammo = int(featurea[2].item())
        blast_strength = int(featurea[3].item())
        can_kick = int(featurea[4].item())
        teammate = int(featurea[5].item())
        enemies = int(featurea[6].item())
        reward = 0
        sagents = sl[4]
        sbombs = sl[2]
        es = epistep.tolist()
        es[1] += 2

        # reward_wood
        if int(action[0].item()) == 5:
            position_bomb = np.array([position0, position1])
            m = position_bomb[0]
            n = position_bomb[1]
            l = blast_strength
            f = [l, l, l, l]  # Scope of flame: up down left right
            bomb_flame = np.zeros_like(bomb.numpy())

            # 判断实体墙或边界是否阻断火焰
            flame_up = np.zeros_like(bomb_flame)
            flame_down = np.zeros_like(bomb_flame)
            flame_left = np.zeros_like(bomb_flame)
            flame_right = np.zeros_like(bomb_flame)
            if m - f[0] < 0:  # 上边界
                f[0] = m
            flame_up[m - f[0]:m, n] = 1
            if m + f[1] > bomb_flame.shape[0] - 1:  # 下边界
                f[1] = bomb_flame.shape[0] - 1 - m
            flame_down[m + 1:m + f[1] + 1, n] = 1
            if n - f[2] < 0:  # 左边界
                f[2] = n
            flame_left[m, n - f[2]:n] = 1
            if n + f[3] > bomb_flame.shape[0] - 1:  # 右边界
                f[3] = bomb_flame.shape[0] - 1 - n
            flame_right[m, n + 1:n + f[3] + 1] = 1

            rigid_0 = flame_up * rigid
            rigid_1 = flame_down * rigid
            rigid_2 = flame_left * rigid
            rigid_3 = flame_right * rigid
            if np.argwhere(rigid_0 == 1).size != 0:  # 上实体墙
                rigid_up = np.max(np.argwhere(rigid_0 == 1)[:, 0][0])
                if rigid_up >= m - f[0]:
                    f[0] = m - rigid_up - 1
            if np.argwhere(rigid_1 == 1).size != 0:  # 下实体墙
                rigid_down = np.min(np.argwhere(rigid_1 == 1)[:, 0][0])
                if rigid_down <= m + f[1]:
                    f[1] = rigid_down - m - 1
            if np.argwhere(rigid_2 == 1).size != 0:  # 左实体墙
                rigid_left = np.max(np.argwhere(rigid_2 == 1)[0, :][1])
                if rigid_left >= n - f[2]:
                    f[2] = n - rigid_left - 1
            if np.argwhere(rigid_3 == 1).size != 0:  # 右实体墙
                rigid_right = np.min(np.argwhere(rigid_3 == 1)[0, :][1])
                if rigid_right <= n + f[3]:
                    f[3] = rigid_right - n - 1
            bomb_flame[m - f[0]:m + f[1] + 1, n] = 1
            bomb_flame[m, n - f[2]:n + f[3] + 1] = 1
            num_wood = np.count_nonzero(wood * bomb_flame == 1)
            reward += num_wood * r_wood
            '''
            # test
            print('rigid')
            print(rigid)
            print('position_bomb')
            print(position_bomb)
            print('f')
            print(f)
            print('l')
            print(l)
            print('bomb_flame')
            print(bomb_flame)
            print('num_wood')
            print(num_wood)
            print('-------------------------------------')
            '''
        return reward

    def epsilon_decay(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.min_epsilon else self.epsilon
