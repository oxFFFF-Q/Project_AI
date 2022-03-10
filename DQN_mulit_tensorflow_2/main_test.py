import time

import constants
import pommerman
import numpy as np

# from DQNAgent_radio import DQNAgent
from DQNAgent_modified import DQNAgent
# from DQNAgent_one_vs_one import DQNAgent
# from DQNAgent_dueling_dqn import DQNAgent
#from DQNAgent_double_dqn import DQNAgent
# from DQNAgent_dueling_dqn import DQNAgent
# from DQNAgent_modified_filter_tong import DQNAgent
# from DQNAgent_radio_filter2 import DQNAgent
# from DQNAgent_noisy import DQNAgent
from pommerman.agents import SimpleAgent, RandomAgent
from utility import featurize2D, reward_shaping


# from DQNAgent_radio import DQNAgent

def main():
    agent1 = DQNAgent()
    agent2 = SimpleAgent()
    agent3 = DQNAgent()
    agent4 = SimpleAgent()

    # agent1 = DQNAgent()
    # agent2 = DQNAgent()
    # agent3 = DQNAgent()
    # agent4 = DQNAgent()

    agent_list = [agent1, agent2, agent3, agent4]

    #env = pommerman.make("PommeFFACompetitionFast-v0", agent_list)
    env = pommerman.make("PommeRadioCompetition-v2", agent_list)

    episode_rewards = []  # 记录平均reward

    win = 0
    draw = 0
    total_game = 0

    total_numOfSteps = 0
    episode = 0
    # while True:
    for i in range(100):
        # agent1.save_model()
        current_state = env.reset()
        # 将state 转化 1D array

        episode_reward = 0
        numOfSteps = 0
        episode += 1
        done = False

        while not done:

            state_feature = featurize2D(current_state[0])

            numOfSteps += 1
            total_numOfSteps += 1
            # if numOfSteps % 10 == 0:
            #     actions = env.act(current_state)
            #     actions[0] = 5
            #     print("BOMB!")
            # if constants.epsilon > np.random.random() and total_numOfSteps >= constants.MIN_REPLAY_MEMORY_SIZE:
            # if constants.epsilon > np.random.random():
            #     # 获取动作
            actions = env.act(current_state)
            actions[0] = np.argmax(agent1.action_choose(state_feature)).tolist()
            #actions[0] = agent1.act_filter(current_state[0], actions[0])

            # else:
            #     # 随机动作
            #     actions = env.act(current_state)
            #     print("simple: ", actions[0])
            #     # actions[0] = random.randint(0, 5)

            new_state, result, done, info = env.step(actions)

            if 10 not in new_state[0]["alive"]:
                done = True

            # reward_shaping
            # agent1.buffer.append_action(actions[0])
            # reward = reward_shaping(current_state[0], new_state[0], actions[0], result[0], agent1.buffer.buffer_action)

            # print("action: ", actions[0], "step_reward: ", reward)
            # print("step reward: ",reward)
            # next_state_feature = featurize2D(new_state[0])
            # episode_reward += reward

            # 每一定局数显示游戏画面
            # if constants.SHOW_PREVIEW and not episode % constants.SHOW_GAME:
            env.render()
            time.sleep(10000)

            # 储存记忆
            # agent1.buffer.append([state_feature, actions[0], reward, next_state_feature, done])

            # 学习!
            # agent1.train()

            # 更新state
            current_state = new_state

            if done:
                break

        result = 0

        if done:
            episode_rewards.append(episode_reward)
            total_game += 1
            if 0 in info.get('winners', []):
                win += 1
                result = 2

        # 记录胜负情况
        if numOfSteps == constants.MAX_STEPS + 1:
            draw += 1
            result = 1

        win_rate = win / total_game
        draw_rate = draw / total_game

        if episode % constants.SHOW_EVERY == 0:
            if result == 1:
                print("{} episodes done, result: {} , steps: {}".format(episode,
                                                                        'draw',
                                                                        numOfSteps))

                print("Reward {:.2f}, Average Episode Reward: {:.3f}, win_rate:{:.2f}, draw_rate:{:.2f}".format(
                    episode_reward,
                    np.mean(episode_rewards),
                    win_rate,
                    draw_rate))
            else:
                print("{} episodes done, result: {} , steps: {}".format(episode,
                                                                        'win' if result == 2 else "lose",
                                                                        numOfSteps))

                print("Reward {:.3f}, Average Episode Reward: {:.3f}, win_rate:{:.2f}, draw_rate:{:.2f}".format(
                    episode_reward,
                    np.mean(episode_rewards),
                    win_rate,
                    draw_rate))
    print("win: ", win, " draw: ", draw)
    env.close()


if __name__ == '__main__':
    main()
