import random

import numpy as np


def action_filter(current_state, action_list):
    """对action进行筛选来提高存活率"""
    X = current_state["position"][0]
    Y = current_state["position"][1]
    # 记录agent周围可移动的位置
    move_list = make_move_list(current_state, X, Y)
    # 若不能移动，则执行action 0（等死）
    if move_list is None:
        return 0
    # 统计执行各种action后 agent的位置
    moveable_position = make_move_position(move_list, X, Y, action_list)
    # 统计可视范围内的炸弹
    bomb_list = make_bomb_list(current_state)
    # 为新位置的危险程度分级
    moveable_position_score = make_dangerous_list(current_state, moveable_position, bomb_list, action_list)

    actions = []
    # 如果存在安全位置，使用安全位置
    for action in moveable_position_score:
        if action[3] == 0:
            actions.append(action[2])
    # 原本动作是安全的，直接选择原本的动作
    if action_list[-1] in actions:
        return action_list[-1]

    # 无安全位置，选择危险度1的位置
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 1:
                actions.append(action[2])
    # 原本动作是安全的，直接选择原本的动作
    if action_list[-1] in actions:
        return action_list[-1]
    # 无危险度1的位置，选择危险度2的位置
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 2:
                actions.append(action[2])
    # 无危险度2的位置，选择危险度3的位置
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 3:
                actions.append(action[2])

    if actions is None:
        return random.randint(0, 5)

    # 若有多种可选动作，优先选择非0的动作
    if len(actions) != 0:
        modified_action = [action for action in actions if action > 0]
        if len(modified_action) > 0:
            modified_action = random.sample(actions, 1)
            return modified_action[0]
        else:
            return 0

    else:
        return action_list[-1]


def make_move_list(current_state, X, Y):
    """检查action是否可执行（有意义）"""

    def check_bomb_action(current_state):
        """检查action 5是否有意义"""
        meaningful = False
        X = current_state["position"][0]
        Y = current_state["position"][1]
        blast_strength = current_state["blast_strength"]
        # 判断炸弹左方是否有墙
        for strength in range(1, blast_strength):
            # 检查是否超出地图边界
            if Y - strength < 0:
                break
            # 如果是rigid，则break
            elif current_state["board"][X][Y - strength] == 1:
                break
            # 检查是否有wood,有则有意义
            elif current_state["board"][X][Y - strength] == 2:
                meaningful = True
                return meaningful
            # 如果爆炸范围内有敌人，则有意义
            elif current_state["board"][X][Y - strength] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # 判断炸弹右方是否有墙
        for strength in range(1, blast_strength):
            # 检查是否超出地图边界
            if Y + strength > 10:
                break
            # 如果是rigid，则break
            elif current_state["board"][X][Y + strength] == 1:
                break
            # 检查是否有wood
            elif current_state["board"][X][Y + strength] == 2:
                meaningful = True
                return meaningful
            # 如果爆炸范围内有敌人，则有意义
            elif current_state["board"][X][Y + strength] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # 判断炸弹上方是否有墙
        for strength in range(1, blast_strength):
            # 检查是否超出地图边界
            if X - strength < 0:
                break
            # 如果是rigid，则break
            elif current_state["board"][X - strength][Y] == 1:
                break
            # 检查是否有wood
            elif current_state["board"][X - strength][Y] == 2:
                meaningful = True
                return meaningful
            # 如果爆炸范围内有敌人，则有意义
            elif current_state["board"][X - strength][Y] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # 判断炸弹下方是否有墙
        for strength in range(1, blast_strength):
            # 检查是否超出地图边界
            if X + strength > 10:
                break
            # 如果是rigid，则break
            elif current_state["board"][X + strength][Y] == 1:
                break
            # 检查是否有wood
            elif current_state["board"][X + strength][Y] == 2:
                meaningful = True
                return meaningful
            # 如果爆炸范围内有敌人，则有意义
            elif current_state["board"][X + strength][Y] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful
        return meaningful

    def check_moveable(current_state, X, Y):
        """检查位置是否可移动"""
        moveable = False
        if X < 0 or X > 10 or Y < 0 or Y > 10:
            return moveable
        elif current_state["board"][X][Y] in [0, 4, 6, 7, 8]:
            moveable = True
            return moveable
        elif current_state["board"][X][Y] == 3 and current_state["can_kick"] is True:
            moveable = True
            return moveable

        return moveable

    move_list = [0]
    can_up = check_moveable(current_state, X - 1, Y)
    if can_up:
        move_list.append(1)
    can_down = check_moveable(current_state, X + 1, Y)
    if can_down:
        move_list.append(2)
    can_left = check_moveable(current_state, X, Y - 1)
    if can_left:
        move_list.append(3)
    can_right = check_moveable(current_state, X, Y + 1)
    if can_right:
        move_list.append(4)
    if current_state["ammo"] != 0 and check_bomb_action(current_state):
        move_list.append(5)
    return move_list


def make_move_position(move_list, X, Y, action_list):
    """计算可移动位置的危险度"""
    # 前两列，移动后位置，第三列，动作，第四列，危险度
    moveable_position = []

    for action in move_list:
        if action == 0:
            moveable_position.append([X, Y, 0, 0])
        elif action == 1:
            moveable_position.append([X - 1, Y, 1, 0])
        elif action == 2:
            moveable_position.append([X + 1, Y, 2, 0])
        elif action == 3:
            moveable_position.append([X, Y - 1, 3, 0])
        elif action == 4:
            moveable_position.append([X, Y + 1, 4, 0])
        elif action == 5:
            if action_list[-2] != 5:
                moveable_position.append([X, Y, 5, 0])
    return moveable_position


def make_bomb_list(current_state):
    bomb_list = []
    for X in range(11):
        for Y in range(11):
            if current_state["board"][X][Y] == 3:
                bomb_list.append([X, Y])

    return bomb_list


def make_dangerous_list(current_state, moveable_position, bomb_list, action_list):
    # safe 0 , dangerous 1, high_dangerous 2, death 3.

    def check_block(current_state, position_agent, position_bomb):
        # 检查雷与agent之间是否有阻碍
        block = False
        if position_agent[0] != position_bomb[0]:
            for index in range(1, abs(position_agent[0] - position_bomb[0])):
                # 检查是否有wood and rigid
                if current_state["board"][min(position_agent[0], position_bomb[0]) + index][position_agent[1]] in [1,
                                                                                                                   2]:
                    block = True
                    break
        elif position_agent[1] != position_bomb[1]:
            for index in range(1, abs(position_agent[1] - position_bomb[1])):
                # 检查是否有wood and rigid
                if current_state["board"][position_agent[0]][min(position_agent[1], position_bomb[1]) + index] in [1,
                                                                                                                   2]:
                    block = True
                    break

        return block

    # 检查死胡同
    def check_dead_end(current_state, moveable_position, action_list):
        for position in moveable_position:
            if action_list[-2] == 5 and position[2] == 1:
                if (position[1] - 1 < 0 or current_state["board"][position[0]][position[1] - 1] in [1, 2, 3]) and \
                        (position[1] + 1 > 10 or current_state["board"][position[0]][position[1] + 1] in [1, 2, 3]) and \
                        (position[0] - 1 < 0 or current_state["board"][position[0] - 1][position[1]] in [1, 2, 3]):
                    position[-1] = 3

            elif action_list[-2] == 5 and position[2] == 2:
                if (position[1] - 1 < 0 or current_state["board"][position[0]][position[1] - 1] in [1, 2, 3]) and \
                        (position[1] + 1 > 10 or current_state["board"][position[0]][position[1] + 1] in [1, 2, 3]) and \
                        (position[0] + 1 > 10 or current_state["board"][position[0] + 1][position[1]] in [1, 2, 3]):
                    position[-1] = 3

            elif action_list[-2] == 5 and position[2] == 3:
                if (position[0] - 1 < 0 or current_state["board"][position[0] - 1][position[1]] in [1, 2, 3]) and \
                        (position[1] - 1 < 0 or current_state["board"][position[0]][position[1] - 1] in [1, 2, 3]) and \
                        (position[0] + 1 > 10 or current_state["board"][position[0] + 1][position[1]] in [1, 2, 3]):
                    position[-1] = 3

            elif action_list[-2] == 5 and position[2] == 4:
                if (position[0] - 1 < 0 or current_state["board"][position[0] - 1][position[1]] in [1, 2, 3]) and \
                        (position[1] + 1 > 10 or current_state["board"][position[0]][position[1] + 1] in [1, 2, 3]) and \
                        (position[0] + 1 > 10 or current_state["board"][position[0] + 1][position[1]] in [1, 2, 3]):
                    position[-1] = 3

        return moveable_position

    # 检查火焰
    def check_flame(current_state, moveable_position):
        for position in moveable_position:
            if current_state["flame_life"][position[0]][position[1]] != 0:
                position[-1] = 3
        return moveable_position

    # bomb_location_all = np.where(np.array(current_state["board"]) == 3)
    # if len(bomb_location_all[0]) == 0:
    #     return moveable_position
    moveable_position = check_dead_end(current_state, moveable_position, action_list)

    # 即使视野里没有炸弹，也要检查是否有火焰
    if len(bomb_list) == 0:
        moveable_position = check_flame(current_state, moveable_position)
        return moveable_position
        # 将moveable position 按危险程度分类，同一位置为多颗炸弹或火焰威胁时，优先记录最高危险级别.
    for agent_position in moveable_position:
        for bomb_position in bomb_list:
            if bomb_position[0] == agent_position[0] or bomb_position[1] == agent_position[1]:
                if check_block(current_state, agent_position, bomb_position) is False and abs(
                        agent_position[0] - bomb_position[0]) + abs(agent_position[1] - bomb_position[1]) <= \
                        (current_state["bomb_blast_strength"][bomb_position[0]][bomb_position[1]] - 1):
                    if current_state["flame_life"][agent_position[0]][agent_position[1]] >= 1 or \
                            current_state["bomb_life"][bomb_position[0]][bomb_position[1]] == 1:
                        agent_position[3] = 3
                    elif 1 < current_state["bomb_life"][bomb_position[0]][bomb_position[1]] <= 3:
                        if agent_position[3] < 2:
                            agent_position[3] = 2
                    elif 4 <= current_state["bomb_life"][bomb_position[0]][bomb_position[1]]:
                        if agent_position[3] < 1:
                            agent_position[3] = 1

    return moveable_position
