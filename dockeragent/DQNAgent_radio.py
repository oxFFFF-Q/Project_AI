from pommerman.agents import BaseAgent
import numpy as np
import tensorflow as tf


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.DQN_model = tf.keras.models.load_model("./first_try")

    def act(self, observation, action_space):
        obs = self.featurize2D(observation)
        state_reshape = tf.reshape(obs, (-1, 18, 11, 11))
        q_table = self.DQN_model.predict_on_batch(state_reshape)
        action = np.argmax(q_table).tolist()
        return action

    def episode_end(self, reward):
        pass

    def shutdown(self):
        pass
    # def save_model(self):
    #     self.DQN_model.save("first_try")

    def featurize2D(self, states):
        feature2D = []
        # 共18个矩阵
        shape = (11, 11)

        # path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4
        def get_matrix(board, key):
            res = board[key]
            return res.reshape(shape).astype(np.float64)

        def get_map(board, item):
            map = np.zeros(shape)
            map[board == item] = 1
            return map

        board = get_matrix(states, "board")

        path = get_map(board, 0)
        rigid = get_map(board, 1)
        wood = get_map(board, 2)
        bomb = get_map(board, 3)
        flame = get_map(board, 4)
        fog = get_map(board, 5)
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

    def rebuild_1D_element(self ,states):
        ammo = states["ammo"]
        ammo_2D = np.full((11, 11), ammo)

        blast_strength = states["blast_strength"]
        blast_strength_2D = np.full((11, 11), blast_strength)

        can_kick = states["can_kick"]
        can_kick_2D = np.full((11, 11), int(can_kick))

        return ammo_2D, blast_strength_2D, can_kick_2D
