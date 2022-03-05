import constants
import pommerman
import numpy as np
import pandas as pd
import random

from DQNAgent_modified import DQNAgent
# from DQNAgent_ddqn_imitation import DQNAgent
# from DQNAgent_ddqn_nstep import DQNAgent
# from DQNAgent_ddqn_pri import DQNAgent
# from DQNAgent_ddqn_nstep_pri import DQNAgent
# from DQNAgent_ddqn_noisy import DQNAgent
# from DQNAgent_ddqn_nstep_noisy import DQNAgent
# from DQNAgent_ddqn_nstep_pri_noisy import DQNAgent
from pommerman.agents import SimpleAgent
from utility_radio import featurize2D, reward_shaping
from communication import message


def main():
    agent1 = DQNAgent()
    agent2 = SimpleAgent()
    agent3 = SimpleAgent()
    agent4 = SimpleAgent()

    agent_list = [agent1, agent2, agent3, agent4]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    #  ['PommeFFACompetition-v0', 'PommeFFACompetitionFast-v0', 'PommeFFAFast-v0',
    #  'PommeFFA-v1', 'OneVsOne-v0', 'PommeRadioCompetition-v2', 'PommeRadio-v2',
    #  'PommeTeamCompetition-v0', 'PommeTeamCompetitionFast-v0', 'PommeTeamCompetition-v1',
    #  'PommeTeam-v0', 'PommeTeamFast-v0']

    episode_rewards = []  # 记录平均reward

    win = 0
    draw = 0
    total_game = 0
    reward_to_csv = []
    result_to_csv = []

    total_numOfSteps = 0
    episode = 0

    while True:

        current_state = env.reset()

        # 将state 转化 1D array

        episode_reward = 0
        numOfSteps = 0
        episode += 1
        done = False

        while not done:
            x = current_state
            mess, current_state[0] = message(current_state[0])
            state_feature = featurize2D(current_state[0])

            numOfSteps += 1
            total_numOfSteps += 1

            # if constants.epsilon > np.random.random() and total_numOfSteps >= constants.MIN_REPLAY_MEMORY_SIZE:
            #     # if constants.epsilon > np.random.random():
            #     # 获取动作
            #     actions = env.act(current_state)
            #     action = np.argmax(agent1.action_choose(state_feature)).tolist()
            #     agent1.buffer.append_action(action)
            #     actions[0] = [action, mess[0], mess[1]]
            # else:
            #     # simple动作
            #     actions = env.act(current_state)
            #     actions[0] = [actions[0], 0, 0]
            #     # actions[0] = random.randint(0, 5)

            actions = env.act(current_state)
            actions[0] = [actions[0], mess[0], mess[1]]
            new_state, result, done, info = env.step(actions)

            if 10 not in new_state[0]["alive"]:
                done = True

            # reward_shaping

            reward = reward_shaping(current_state[0], new_state[0], agent1.buffer.buffer_action[0], result[0],
                                    agent1.buffer.buffer_action)
            # print("reward: ",reward)
            next_state_feature = featurize2D(new_state[0])
            episode_reward += reward

            # 每一定局数显示游戏画面
            # if constants.SHOW_PREVIEW and not episode % constants.SHOW_GAME:
            # env.render()

            # ddqn, ddqnnoisy
            agent1.buffer.append((state_feature, actions[0][0], reward, next_state_feature, done))
            agent1.train()

            # ddqn_nstep, ddqn_nstep_noisy
            # mark_nstep = agent1.buffer.append_nstep(state_feature, actions[0], reward, next_state_feature, done)
            # if mark_nstep:
            #      agent1.train()

            # ddqn_pri, double_pri, pri
            # td_error = agent1.calculate_td_error(state_feature, actions[0], reward, next_state_feature, done)
            # agent1.buffer.append_pri(state_feature, actions[0], reward, next_state_feature, done, td_error)
            # agent1.train()

            # ddqn_nstep_pri
            # td_error = agent1.calculate_td_error(state_feature, actions[0], reward, next_state_feature, done)
            # mark_nstep = agent1.buffer.append_nstep_pri(state_feature, actions[0], reward, next_state_feature, done, td_error)
            # if mark_nstep:
            #     agent1.train()

            # ddqn_nstep_pri, ddqn_nstep_pri_noisy
            # td_error = agent1.calculate_td_error(state_feature, actions[0], reward, next_state_feature, done)
            # mark_nstep = agent1.buffer.append_nstep_pri(state_feature, actions[0], reward, next_state_feature, done, td_error)
            # if mark_nstep:
            #       agent1.train()

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
        # 存reward
        reward_to_csv.append(episode_reward)
        # 存result
        result_to_csv.append(result)

        if episode % constants.SHOW_EVERY == 0:
            if result == 1:
                print("{} episodes done, result: {} , steps: {}".format(episode,
                                                                        'draw',
                                                                        numOfSteps))

                print("Reward {:.2f}, Average Episode Reward: {:.3f}, win_rate:{:.2f}, draw_rate:{:.2f}"
                    .format(
                    episode_reward,
                    np.mean(episode_rewards),
                    win_rate,
                    draw_rate))
            else:
                print("{} episodes done, result: {} , steps: {}".format(episode,
                                                                        'win' if result == 2 else "lose",
                                                                        numOfSteps))

                print("Reward {:.3f}, Average Episode Reward: {:.3f}, win_rate:{:.2f}, draw_rate:{:.2f}"
                    .format(
                    episode_reward,
                    np.mean(episode_rewards),
                    win_rate,
                    draw_rate))

        # agent1.epsilon_decay()

        agent1.save_weights(episode)

        # 记录结果，留作图表
        if episode % 100 == 0:
            df_reward = pd.DataFrame({"reward": reward_to_csv})
            df_reward.to_csv("reward.csv", index=False, mode="a", header=False)
            print("successfully saved reward")
            reward_to_csv = []
            df_result = pd.DataFrame({"result": result_to_csv})
            df_result.to_csv("result.csv", index=False, mode="a", header=False)
            print("successfully saved result")
            result_to_csv = []

    env.close()


if __name__ == '__main__':
    main()