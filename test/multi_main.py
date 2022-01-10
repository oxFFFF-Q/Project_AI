import collections
import pommerman
import gym
import torch
import argparse
import random
import numpy as np
import os

from pommerman import agents
from pommerman.configs import radio_v2_env, team_v0_fast_env, radio_competition_env
from DQN2Agent import DQN2Agent
from utils import featurize, CustomEnvWrapper, featurize2
from plot import plot_win_rate


def main():
    """解析参数"""
    parser = argparse.ArgumentParser(description='DQN pommerman MARL')
    parser.add_argument('--episodes', type=int, default=1000, help='episodes')
    parser.add_argument('--maxsteps', type=int, default=200, help='maximum steps')
    parser.add_argument('--showevery', type=int, default=1, help='report loss every n episodes')

    parser.add_argument('--epsilon', type=float, default=0.9, help='parameter for epsilon greedy')
    parser.add_argument('--eps_decay', type=float, default=0.9999, help='epsilon decay rate')
    parser.add_argument('--min_eps', type=float, default=0.05, help='minimum epsilon for decaying')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay rate')
    parser.add_argument('--lr_decay_s', type=float, default=100, help='learning rate decay rate setp size')

    parser.add_argument('--capacity', type=int, default=100000, help='capacity for replay buffer')
    parser.add_argument('--batch', type=int, default=128, help='batch size for replay buffer')
    parser.add_argument('--tryepi', type=int, default=100, help='episode for agent to gain experience')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number')
    parser.add_argument('--win_in_epi', type=int, default='200', help='calculate win in epi..')
    parser.add_argument('--ranepi', type=int, default='50', help='agent go random action in epi..')
    args = parser.parse_args()

    # GPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("GPU using status: ", args.device)

    agent_list = [agents.SimpleAgent(), agents.SimpleAgent()]  # placeholder
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    agent1 = DQN2Agent(env, args)  # TODO: assertionerror; not agents.BaseAgent??
    agent2 = agents.SimpleAgent()
    # agent3 = DQN2Agent(env, args)
    agent3 = agents.RandomAgent()
    agent4 = agents.SimpleAgent()

    agent_list = [agent1, agent2, agent3, agent4]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)

    # plot
    list_win = []

    episode_rewards = []
    action_n = 6

    # 加载模型
    agent1.load_model()
    # agent3.load_model()
    # collect win times
    # if os.path.exists('model_dqn2.pt'):
    #     args.epsilon = 0.1
    #     args.eps_decay = 0.98
    #     args.tryepi = 0
    #     args.ranepi = 0

    win_buffer = collections.deque(maxlen=args.win_in_epi)
    for episode in range(args.episodes):
        args.episode = episode + 1
        # 固定地图
        random.seed(2)
        np.random.seed(2)

        states = env.reset()
        # print('epi:', episode)
        done = False
        episode_reward = 0
        die = 0

        # lr_decay
        if (episode + 1) % args.lr_decay_s == 0:
            epi = episode + 1
            if episode + 1 == args.episodes:
                epi = -1
            agent1.lrdecay(epi)

        for step in range(args.maxsteps):
            state_feature1 = featurize2(env, states[0])
            # state_feature3 = featurize2(env, states[2])

            # 刷新环境
            # if episode % 100 == 0 and episode != 0:
            #     env.render()
            if os.path.exists('model_dqn2.pt'):
                env.render()
            else:
                if args.episode > (args.episodes - 10):
                    env.render()

            # 选择action
            actions = env.act(states)
            if step%10 == 0:
                actions[0] = 5
            else:
                actions[0] = agent1.choose_action(state_feature1)

            next_state, reward, done, info = env.step(actions)  # n-array with action for each agent
            if 10 not in next_state[0]['alive']:
                info['winners'] = [1, 3]
                reward = [-1, 1, -1, 1]

            next_state_feature1 = featurize2(env, next_state[0])

            if 10 not in next_state[0]['alive']:
                info['winners'] = [1, 3]
                reward = [-1, 1, -1, 1]
            next_state_feature1 = featurize2(env, next_state[0])
            # next_state_feature3 = featurize2(env, next_state[2])
            # episode_reward += reward[0]

            # 存储记忆
            agent1.buffer.append([state_feature1, actions[0], reward[0], next_state_feature1, done])
            # agent3.buffer.append([state_feature3, actions[2], reward[2], next_state_feature3, done])
            # 先走batch步之后再开始学习
            if agent1.buffer.size() >= args.batch and episode > 50:
                # if agent1.buffer.size() >= args.batch:
                agent1.update(args.gamma, args.batch)
            # if episode > args.tryepi and agent1.buffer.size() >= args.batch:
            #     agent3.update(args.gamma, args.batch)
            # 更新state
            states = next_state

            if done:
                break
            # agent1 die -> game over
            if 10 not in next_state[0]['alive']:
                break

        # if done:
        #    episode_rewards.append(episode_reward)
        if episode % args.showevery == 0:
            if 0 in info.get('winners', []):
                print(f"Episode: {episode + 1:2d} finished, result: Win")
            elif 1 in info.get('winners', []):
                print(f"Episode: {episode + 1:2d} finished, result: Lose")
            else:
                print(f"Episode: {episode + 1:2d} finished, result: Not finish")
            # print(f"Avg Episode Reward: {np.mean(episode_rewards)}")

        if episode > args.win_in_epi:

            if 0 in info.get('winners', []):
                win_buffer.append(1)
            elif 1 in info.get('winners', []):
                win_buffer.append(0)
            if len(win_buffer) == args.win_in_epi:
                avg = sum(win_buffer) / len(win_buffer)
                print(f"current winrate: {avg}")
                list_win.append(avg)
                if len(list_win) % 500 == 0:
                    plot_win_rate(list_win)

        # agent3.epsdecay()

    agent1.save_model()
    # agent3.save_model()
    env.close()

    # TODO: Implement Target Network


if __name__ == '__main__':
    main()