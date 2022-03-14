from Group_C.utility import constants
from agents.Agent1 import DQNAgent as agent_1
from agents.Agent3 import DQNAgent as agent_3
from pommerman.agents import SimpleAgent

import pommerman


def main():
    agent1 = agent_1()
    agent2 = SimpleAgent()
    agent3 = agent_3()
    agent4 = SimpleAgent()

    agent_list = [agent1, agent2, agent3, agent4]
    env = pommerman.make("PommeRadioCompetition-v2", agent_list)
    # Record average reward
    episode_rewards = []
    win = 0
    draw = 0
    total_game = 0

    total_numOfSteps = 0
    episode = 0
    # Conduct 100 rounds of testing
    for i in range(100):
        current_state = env.reset()
        # Convert state to 1D array
        episode_reward = 0
        numOfSteps = 0
        episode += 1
        done = False

        while not done:

            numOfSteps += 1
            total_numOfSteps += 1
            actions = env.act(current_state)
            env.render()
            new_state, result, done, info = env.step(actions)

            if 10 not in new_state[0]["alive"]:
                done = True

            # Update state
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

        # Record wins and losses
        if numOfSteps == constants.MAX_STEPS + 1:
            draw += 1
            result = 1

        win_rate = win / total_game
        draw_rate = draw / total_game

        if episode % constants.SHOW_EVERY == 0:
            if result == 1:
                print("{} episodes done, result: {} , steps: {}, win_rate:{:.2f}, draw_rate:{:.2f}".format(episode,
                                                                                                           'draw',
                                                                                                           numOfSteps,
                                                                                                           win_rate,
                                                                                                           draw_rate))
            else:
                print("{} episodes done, result: {} , steps: {}, win_rate:{:.2f}, draw_rate:{:.2f}".format(episode,
                                                                                                           'win' if result == 2 else "lose",
                                                                                                           numOfSteps,
                                                                                                           win_rate,
                                                                                                           draw_rate))

    print("win: ", win, " draw: ", draw)
    env.close()


if __name__ == '__main__':
    main()
