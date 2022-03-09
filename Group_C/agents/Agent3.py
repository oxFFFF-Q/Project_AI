from pommerman.agents import BaseAgent
from action_filter import action_filter
import numpy as np
import tensorflow as tf
import collections
import os


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.gpu = os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.DQN_model = tf.keras.models.load_model("./Agent3")
        self.action_list = collections.deque([0, 0, 0], maxlen=3)

    def action_predict(self, state):
        state_feature = self.featurize2D(state)
        state_reshape = tf.reshape(state_feature, (-1, 18, 11, 11))
        q_table = self.DQN_model.predict(state_reshape)
        return q_table

    def act(self, observation, action_space):
        action = np.argmax(self.action_predict(observation)).tolist()
        self.action_list.append(action)
        action = action_filter(observation, self.action_list)
        self.action_list[-1] = action
        return action

    def episode_end(self, reward):
        pass

    def shutdown(self):
        pass

    def featurize2D(self, states, partially_obs=True):
        # 共18个矩阵
        X = states["position"][0]
        Y = states["position"][1]
        shape = (11, 11)

        # path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4
        def get_partially_obs(states, X, Y):
            # board = np.zeros(shape)
            board = np.full(shape, 5)
            for x in range(10):
                for y in range(10):
                    if X - 4 <= x <= X + 4 and Y - 4 <= y <= Y + 4:
                        board[x][y] = states["board"][x][y]
            states["board"] = board
            return states

        def get_matrix(board, key):
            res = board[key]
            return res.reshape(shape).astype(np.float64)

        def get_map(board, item):
            map = np.zeros(shape)
            map[board == item] = 1
            return map

        if partially_obs:
            states = get_partially_obs(states, X, Y)

        board = get_matrix(states, "board")

        path = get_map(board, 0)
        rigid = get_map(board, 1)
        wood = get_map(board, 2)
        bomb = get_map(board, 3)
        flame = get_map(board, 4)
        fog = np.zeros(shape)
        agent1 = get_map(board, 10)
        agent2 = get_map(board, 11)
        agent3 = get_map(board, 12)
        agent4 = get_map(board, 13)

        power_up = []
        for row in board:
            new_row = []
            for num in row:
                if num == 6 or num == 7 or num == 8:
                    new_row.append(1)
                else:
                    new_row.append(0.0)
            power_up.append(new_row)

        bomb_blast_strength = get_matrix(states, 'bomb_blast_strength')
        bomb_life = get_matrix(states, 'bomb_life')
        bomb_moving_direction = get_matrix(states, 'bomb_moving_direction')
        flame_life = get_matrix(states, 'flame_life')

        ammo_2D, blast_strength_2D, can_kick_2D = self.rebuild_1D_element(states)

        feature2D = [path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4, bomb_blast_strength,
                     bomb_life, bomb_moving_direction, flame_life, ammo_2D, blast_strength_2D, can_kick_2D]

        return np.array(feature2D)

    def rebuild_1D_element(self, states):
        shape = (11, 11)
        ammo = states["ammo"]
        ammo_2D = np.full(shape, ammo)

        blast_strength = states["blast_strength"]
        blast_strength_2D = np.full(shape, blast_strength)

        can_kick = states["can_kick"]
        can_kick_2D = np.full(shape, int(can_kick))

        return ammo_2D, blast_strength_2D, can_kick_2D
