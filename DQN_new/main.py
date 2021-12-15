import random
import constants
import pommerman
import numpy as np

from DQNAgent import DQNAgent
from pommerman.agents import SimpleAgent
from utility import featurize2D


def main():

    agent1 = DQNAgent()
    agent2 = SimpleAgent()

    agent_list = [agent1, agent2]
    env = pommerman.make('OneVsOne-v0', agent_list)

    episode_rewards = []  # 记录平均reward
    action_n = env.action_space.n  # action空间

    win = 0
    total_game = 0
    win_rate = 0

    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)

    for episode in range(constants.EPISODES + 1):

        current_state = env.reset()
        # 将state 转化 1D array
        state_feature = featurize2D(current_state)
        episode_reward = 0

        done = False
        while not done:
            if constants.epsilon > np.random.random():
                # 获取动作
                actions = env.act(current_state)
                actions[0] = np.argmax(agent1.get_q_value(state_feature))
            else:
                # 随机动作
                actions = env.act(current_state)
                actions[0] = random.randint(0, 6)

            new_state, reward, done, info = env.step(actions)
            next_state_feature = featurize2D(new_state)
            episode_reward += reward[0]

            # 每一定局数显示游戏画面
            # if constants.SHOW_PREVIEW and not episode % constants.SHOW_GAME:
            #     env.render()

            # 储存记忆
            agent1.buffer.append((state_feature, actions, reward, next_state_feature, done))

            # 学习!
            agent1.train(done, episode)

            # 更新state
            current_state = new_state

            if done:
                break

        if done:
            episode_rewards.append(episode_reward)
            total_game += 1
            if 0 in info.get('winners', []):
                win += 1

        win_rate = win / total_game

        if episode % constants.SHOW_EVERY == 0:
            print("{} of 3000 episodes done, result: {}".format(episode + 1,
                                                                'Win' if 0 in info.get('winners', []) else 'Lose'))
            print("Average Episode Reward: {:.3f}, win_rate_last_1000_game:{:.2f}".format(np.mean(episode_rewards),
                                                                                          win_rate))

        if total_game >= 999:
            win = 0
            total_game = 0

        agent1.epsilon_decay()


    env.close()


if __name__ == '__main__':
    main()
