import pommerman
import gym
import torch
import argparse
import random
import numpy as np
import collections


from pommerman import agents
from pommerman.configs import one_vs_one_env
from DQNAgent import DQNAgent
from utils import featurize
import os



def main():
    """解析参数"""
    parser = argparse.ArgumentParser(description='DQN pommerman MARL')
    parser.add_argument('--episodes', type=int, default=5000, help='episodes')
    parser.add_argument('--maxsteps', type=int, default=200, help='maximum steps')
    parser.add_argument('--showevery', type=int, default=1, help='report loss every n episodes')

    parser.add_argument('--epsilon', type=float, default=0.9, help='parameter for epsilon greedy')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay rate')
    parser.add_argument('--min_eps', type=float, default=0.05, help='minimum epsilon for decaying')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--capacity', type=int, default=100000, help='capacity for replay buffer')
    parser.add_argument('--batch', type=int, default=201, help='batch size for replay buffer')
    parser.add_argument('--tryepi', type=int, default=50, help='episode for agent to gain experience')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number')
    parser.add_argument('--win_in_epi', type=int, default='50', help='calculate win in epi..')
    parser.add_argument('--ranepi', type=int, default='2000', help='agent go random action in epi..')
    args = parser.parse_args()

    # GPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("GPU using status: ", args.device)

    agent_list = [agents.SimpleAgent(), agents.SimpleAgent()]  # placeholder
    env = pommerman.make('OneVsOne-v0', agent_list)

    agent1 = DQNAgent(env, args)  # TODO: assertionerror; not agents.BaseAgent??
    agent2 = agents.SimpleAgent()

    agent_list = [agent1, agent2]
    env = pommerman.make('OneVsOne-v0', agent_list)

    #episode_rewards = []
    #action_n = env.action_space.n

    # 加载模型
    agent1.load_model()
    # collect win times
    if os.path.exists('model_dqn.pt'):
        args.tryepi = 0
        args.ranepi = 0
        args.epsilon = 0.1

    win_buffer = collections.deque(maxlen=args.win_in_epi)
    for episode in range(args.episodes):
        states = env.reset()
        done = False
        #episode_reward = 0
        step = 0
        for step in range(args.maxsteps):
            state_feature = featurize(env, states[0])
            # 刷新环境
            if episode % 100 == 0 and episode != 0:
                env.render()

            # 选择action
            if (episode < args.tryepi) or (args.epsilon > random.random()):
                actions = env.act(states)
            elif episode < args.ranepi and args.epsilon > random.random():
                actions = env.act(states)
            elif args.epsilon > random.random():
                actions = env.act(states)
                actions[0] = random.randrange(0,6,1)
            else:
                actions = env.act(states)
                dqn_action = agent1.dqnact(state_feature)
                actions[0] = int(np.int64(dqn_action))
                #print(actions[0])

            
            next_state, reward, done, info = env.step(actions)  # n-array with action for each agent

            next_state_feature = featurize(env, next_state[0])
            #episode_reward += reward[0]
            # 存储记忆
            agent1.buffer.append([state_feature, actions[0], reward[0], next_state_feature, done])
            
            # 先走batch步之后再开始学习
            if episode > args.tryepi and agent1.buffer.size() >= args.batch:
                agent1.update(args.gamma, args.batch)
            
            # 更新state
            states = next_state
            
            if done:
                break
        '''
        if done:
            episode_rewards.append(episode_reward)
        '''
        if episode % args.showevery == 0:
            if 0 in info.get('winners', []):
                print(f"Episode: {episode + 1:2d} finished, result: Win")
            elif not done:
                print(f"Episode: {episode + 1:2d} finished, result: Not finish")
            else:
                print(f"Episode: {episode + 1:2d} finished, result: Lose")
            #print(f"Avg Episode Reward: {np.mean(episode_rewards)}")
        
        
        if episode > args.tryepi:
            agent1.epsdecay()
            if 0 in info.get('winners', []):
                win_buffer.append(1)
            elif 1 in info.get('winners', []):
                win_buffer.append(0)
            if len(win_buffer) == args.win_in_epi:
                avg = sum(win_buffer) / len(win_buffer)
                print(f"current winrate: {avg}")




        print('epsilon',agent1.epsilon)

    agent1.save_model()    #保存模型

    env.close()

    # TODO: Implement Target Network


if __name__ == '__main__':
    main()
