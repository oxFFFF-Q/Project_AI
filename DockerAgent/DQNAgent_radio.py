from pommerman.agents import BaseAgent
from pommerman import characters
from utility import featurize2D
import numpy as np
import tensorflow as tf


class DQNAgent(BaseAgent):
    """DQN second try with keras"""

    def __init__(self, character=characters.Bomber):
        super(DQNAgent, self).__init__(character)
        self.DQN_model = tf.keras.models.load_model("model/first_model.h5")

    def act(self, obs, action_space):
        obs = featurize2D(obs)
        state_reshape = tf.reshape(obs, (-1, 18, 11, 11))
        action = np.max(self.DQN_model.predict_on_batch(state_reshape)).tolist()
        return action
