from Group_C.utility import constants
import pommerman
import numpy as np
import pandas as pd
import random

# from DQNAgent_modified import DQNAgent
# from DQNAgent_noisy import DQNAgent
# from DQNAgent_one_vs_one import DQNAgent
# from DQNAgent_dueling_dqn import DQNAgent
# from DQNAgent_double_dqn import DQNAgent
# from DQNAgent_ddqn2 import DQNAgent

# from DQNAgent_rainbow import DQNAgent
from pommerman.agents import SimpleAgent
from Group_C.utility.utility import featurize2D, reward_shaping


def main(strategy='DQN_basic'):
    # strategies: 'DQN_basic', 'DQN_double', 'DQN_dueling', 'DQN_priority', 'DQN_noisy', 'DQN_multi_steps', 'DQN_final'

    if strategy == 'DQN_basic':
        from agents.DQNAgent_basic import DQNAgent
    elif strategy == 'DQN_double':
        from agents.DQNAgent_double_dqn import DQNAgent
    elif strategy == 'DQN_dueling':
        from agents.DQNAgent_dueling_dqn import DQNAgent
    elif strategy == 'DQN_priority':
        from agents.DQNAgent_priority_memory import DQNAgent
    elif strategy == 'DQN_noisy':
        from agents.DQNAgent_noisy import DQNAgent
    elif strategy == 'DQN_multi_steps':
        from agents.DQNAgent_multi_steps import DQNAgent
    elif strategy == 'DQN_final':
        from agents.DQNAgent_final import DQNAgent

    agent1 = DQNAgent()
    agent2 = SimpleAgent()
    agent3 = SimpleAgent()
    agent4 = SimpleAgent()

    agent_list = [agent1, agent2, agent3, agent4]

    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
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

            state_feature = featurize2D(current_state[2])
            numOfSteps += 1
            total_numOfSteps += 1
            # 使用random action收集数据
            if constants.epsilon > np.random.random() and total_numOfSteps >= constants.MIN_REPLAY_MEMORY_SIZE:
                # 获取动作
                actions = env.act(current_state)
                actions[0] = np.argmax(agent1.action_choose(state_feature)).tolist()
            else:
                # 随机动作收集数据
                actions = env.act(current_state)
                actions[0] = random.randint(0, 5)

            new_state, result, done, info = env.step(actions)

            # 若我们的agent死亡，停止本局游戏，加速训练
            if 10 not in new_state[0]["alive"]:
                done = True

            # reward_shaping
            agent1.buffer.append_action(actions[0])
            reward = reward_shaping(current_state[0], new_state[0], actions[0], result[0], agent1.buffer.buffer_action)

            next_state_feature = featurize2D(new_state[0])
            episode_reward += reward

            # 每一定局数显示游戏画面
            if constants.SHOW_PREVIEW and not episode % constants.SHOW_GAME:
                env.render()

            # 储存记忆
            agent1.save_buffer(state_feature, actions[0], reward, next_state_feature, done)
            # 学习!
            agent1.train()
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

        # agent1.epsilon_decay()
        agent1.save_weights(episode)
        # agent1.data_processing(numOfSteps, episode_reward, result, episode)

        # 记录结果，留作图表
        if episode % 50 == 0:
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
    main(strategy='DQN_basic')
    # strategies: 'DQN_basic', 'DQN_double', 'DQN_dueling', 'DQN_priority', 'DQN_noisy', 'DQN_multi_steps', 'DQN_final'
