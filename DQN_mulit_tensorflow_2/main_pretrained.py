import random
import constants
import pommerman
import numpy as np
import pandas as pd

# from DQNAgent import DQNAgent
from DQNAgent_modified_pre import DQNAgent
from pommerman.agents import SimpleAgent
from utility import featurize2D, reward_shaping


def main():
    agent1 = DQNAgent()
    agent2 = SimpleAgent()
    agent3 = SimpleAgent()
    agent4 = SimpleAgent()

    agent_list = [agent1, agent2, agent3, agent4]
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

    dataset_states = np.load("pre_trained/FFA-Dataset-states.npy", allow_pickle=True)
    dataset_actions = np.load("pre_trained/FFA-Dataset-actions1.npy", allow_pickle=True)

    agent1.train(dataset_states, dataset_actions)
    agent1.save_weights()
    # numOfSteps = 0
    # numOfEpisodes = 0
    # while True:
    #     if numOfSteps >= constants.MAX_BUFFER_SIZE_PRE:
    #         break
    #
    #     current_state = env.reset()
    #     # 将state 转化 1D array
    #     state_feature = featurize2D(current_state[0])
    #
    #     done = False
    #     while not done:
    #
    #         numOfSteps += 1
    #
    #         actions = env.act(current_state)
    #
    #         new_state, reward, done, info = env.step(actions)
    #
    #         if 10 not in new_state[0]["alive"]:
    #             done = True
    #
    #         next_state_feature = featurize2D(new_state[0])
    #
    #         # 储存记忆
    #         agent1.buffer.append_pre([state_feature, actions[0]])
    #
    #         # 更新state
    #         current_state = new_state
    #
    #         if done:
    #             numOfEpisodes += 1
    #             print("episode: {:} done, total steps: {:}".format(numOfEpisodes, numOfSteps))
    #             break
    #
    # agent1.save_data()

    env.close()


if __name__ == '__main__':
    main()
