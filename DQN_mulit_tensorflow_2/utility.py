import numpy as np


def reward_shaping(current_state, new_state, action, result, action_list):
    r_win = 0
    r_lose = -1

    r_wood = 0.01
    r_powerup = 0.05
    r_kick = 0.02

    r_lay_bomb = -0.005
    r_lay_bomb_near_enemy = 0.2
    r_attack_teammate = -0.1
    r_get_away_from_bomb = 0.005
    r_get_close_to_bomb = -0.01

    r_avoid = 0.001
    r_move = 0.001
    r_stay = -0.003
    r_move_towards_wood = -0.01
    r_move_loop = -0.005
    r_dead_end = -0.1
    r_ignore_penalty = -0.0015

    reward = 0

    X = current_state["position"][0]
    Y = current_state["position"][1]
    new_X = new_state["position"][0]
    new_Y = new_state["position"][1]

    # 左下情况
    enemies = [11, 12, 13]
    teammate = [10]

    current_grids = []
    if X - 1 >= 0:
        current_grids.append(current_state["board"][X - 1][Y])
    if X + 1 <= 10:
        current_grids.append(current_state["board"][X + 1][Y])
    if Y - 1 >= 0:
        current_grids.append(current_state["board"][X][Y - 1])
    if Y + 1 <= 10:
        current_grids.append(current_state["board"][X][Y + 1])

    # 达到最大步数，为平局
    # if steps == constants.MAX_STEPS + 1:
    #     return reward

    if result == 1:
        reward += r_win

    # 检查是否踢了炸弹
    if current_state["can_kick"] is True and new_state["can_kick"] is False:
        reward += r_kick

    if action == 0:
        reward = check_avoid_flame(reward, r_avoid, current_grids)
        reward = check_corner_bomb(current_state, X, Y, reward, r_avoid, r_stay, current_grids)
        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        return reward

    # 移动reward
    if action == 1:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_move_towards_wood
        else:
            reward += r_move
            reward = check_dead_end(new_state, new_X, new_Y, action_list, reward, r_dead_end)
            reward = check_wood_and_power(current_state, new_X, new_Y, action_list, current_grids, reward,
                                          r_ignore_penalty)
        reward = check_move_loop(action_list, reward, r_move_loop)
        reward = check_power_up(new_X, new_Y, current_state, reward, r_powerup)
        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        return reward

    if action == 2:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_move_towards_wood
        else:
            reward += r_move
            reward = check_dead_end(new_state, new_X, new_Y, action_list, reward, r_dead_end)
            reward = check_wood_and_power(current_state, new_X, new_Y, action_list, current_grids, reward,
                                          r_ignore_penalty)
        reward = check_move_loop(action_list, reward, r_move_loop)
        reward = check_power_up(new_X, new_Y, current_state, reward, r_powerup)
        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        return reward

    if action == 3:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_move_towards_wood
        else:
            reward += r_move
            reward = check_dead_end(new_state, new_X, new_Y, action_list, reward, r_dead_end)
            reward = check_wood_and_power(current_state, new_X, new_Y, action_list, current_grids, reward,
                                          r_ignore_penalty)
        reward = check_move_loop(action_list, reward, r_move_loop)
        reward = check_power_up(new_X, new_Y, current_state, reward, r_powerup)
        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        return reward

    if action == 4:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_move_towards_wood
        else:
            reward += r_move
            reward = check_dead_end(new_state, new_X, new_Y, action_list, reward, r_dead_end)
            reward = check_wood_and_power(current_state, new_X, new_Y, action_list, current_grids, reward,
                                          r_ignore_penalty)
        reward = check_move_loop(action_list, reward, r_move_loop)
        reward = check_power_up(new_X, new_Y, current_state, reward, r_powerup)
        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        return reward

    # 放炸弹reward，包括检查wood， 敌人等
    if action == 5:

        reward = check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y)
        reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb,
                                          r_get_close_to_bomb, action_list)
        if current_state["ammo"] != 0:
            reward += r_lay_bomb
            reward = check_bomb_reward(current_state, X, Y, reward, r_wood, r_lay_bomb_near_enemy, r_attack_teammate,
                                       enemies, teammate)
        else:
            reward += (2 * r_move_towards_wood)

        return reward


def check_bomb_reward(current_state, X, Y, reward, r_wood, r_lay_bomb_near_enemy, r_attack_teammate, enemies, teammate):
    blast_strength = current_state["blast_strength"]
    # 判断炸弹左方是否有墙
    for strength in range(1, blast_strength):
        # 检查是否超出地图边界
        if Y - strength < 0:
            break
        # 检查是否有wood
        elif current_state["board"][X][Y - strength] == 2:
            reward += r_wood
            break
        # 如果是rigid，则break
        elif current_state["board"][X][Y - strength] == 1:
            break
        # 如果爆炸范围内有敌人，获得reward
        elif current_state["board"][X][Y - strength] in enemies:
            reward += r_lay_bomb_near_enemy
        elif current_state["board"][X][Y - strength] in teammate:
            reward += r_attack_teammate

    # 判断炸弹右方是否有墙
    for strength in range(1, blast_strength):
        # 检查是否超出地图边界
        if Y + strength > 10:
            break
        # 检查是否有wood
        elif current_state["board"][X][Y + strength] == 2:
            reward += r_wood
            break
        # 如果是rigid，则break
        elif current_state["board"][X][Y + strength] == 1:
            break
        # 如果爆炸范围内有敌人，获得reward
        elif current_state["board"][X][Y + strength] in enemies:
            reward += r_lay_bomb_near_enemy
        elif current_state["board"][X][Y + strength] in teammate:
            reward += r_attack_teammate

    # 判断炸弹上方是否有墙
    for strength in range(1, blast_strength):
        # 检查是否超出地图边界
        if X - strength < 0:
            break
        # 检查是否有wood
        elif current_state["board"][X - strength][Y] == 2:
            reward += r_wood
            break
        # 如果是rigid，则break
        elif current_state["board"][X - strength][Y] == 1:
            break
        # 如果爆炸范围内有敌人，获得reward
        elif current_state["board"][X - strength][Y] in enemies:
            reward += r_lay_bomb_near_enemy
        elif current_state["board"][X - strength][Y] in teammate:
            reward += r_attack_teammate

    # 判断炸弹下方是否有墙
    for strength in range(1, blast_strength):
        # 检查是否超出地图边界
        if X + strength > 10:
            break
        # 检查是否有wood
        elif current_state["board"][X + strength][Y] == 2:
            reward += r_wood
            break
        # 如果是rigid，则break
        elif current_state["board"][X + strength][Y] == 1:
            break
        # 如果爆炸范围内有敌人，获得reward
        elif current_state["board"][X + strength][Y] in enemies:
            reward += r_lay_bomb_near_enemy
        elif current_state["board"][X + strength][Y] in teammate:
            reward += r_attack_teammate
    return reward


def check_in_flame(current_state, new_state, reward, r_lose, X, Y, new_X, new_Y):
    # 若agent与火焰位置重叠，则死亡，返回reward

    if current_state["flame_life"][X][Y] == 0 and new_state["flame_life"][new_X][new_Y] != 0:
        reward += r_lose
    return reward


def check_power_up(new_X, new_Y, current_state, reward, r_power_up):
    if current_state["board"][new_X][new_Y] in [6, 7, 8]:
        reward += r_power_up

    return reward


def check_corner_bomb(current_state, X, Y, reward, r_avoid, r_stay, current_grids):
    # action 0 来躲避左上bomb
    find_bomb = False
    if X - 1 >= 0 and Y - 1 >= 0 and current_state["board"][X - 1][Y - 1] == 3:
        reward += r_avoid
        find_bomb = True
    if X - 1 >= 0 and Y - 2 >= 0 and current_state["board"][X - 1][Y - 2] == 3:
        reward += r_avoid
        find_bomb = True
        # 右上
    if X - 1 >= 0 and Y + 1 <= 10 and current_state["board"][X - 1][Y + 1] == 3:
        reward += r_avoid
        find_bomb = True
    if X - 1 >= 0 and Y + 2 <= 10 and current_state["board"][X - 1][Y + 2] == 3:
        reward += r_avoid
        find_bomb = True
    # 左下
    if X + 1 <= 10 and Y - 1 >= 0 and current_state["board"][X + 1][Y - 1] == 3:
        reward += r_avoid
        find_bomb = True
    if X + 2 <= 10 and Y - 1 >= 0 and current_state["board"][X + 2][Y - 1] == 3:
        reward += r_avoid
        find_bomb = True
    # 右下
    if X + 1 <= 10 and Y + 1 <= 10 and current_state["board"][X + 1][Y + 1] == 3:
        reward += r_avoid
        find_bomb = True
    if X + 1 <= 10 and Y + 2 <= 10 and current_state["board"][X + 1][Y + 2] == 3:
        reward += r_avoid
        find_bomb = True
    if not find_bomb and 3 not in current_grids:
        reward += r_stay

    return reward


def check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb, r_get_close_to_bomb,
                             action_list):
    # 远离上下左右四个方向的炸弹
    if action_list[2] == 5 and (X != new_X or Y != new_Y):
        reward += r_get_away_from_bomb
    elif action_list[2] == 5 and (X == new_X and Y == new_Y):
        reward += 2 * r_get_close_to_bomb
    # 上
    if X - 1 >= 0 and current_state["board"][X - 1][Y] == 3 and (abs((X - 1) - new_X) + abs(Y - new_Y)) > 1:
        reward += r_get_away_from_bomb
    if X - 1 >= 0 and current_state["board"][X - 1][Y] == 3 and (abs((X - 1) - new_X) + abs(Y - new_Y)) == 1:
        reward += 2 * r_get_close_to_bomb
    if X - 2 >= 0 and current_state["board"][X - 2][Y] == 3 and (abs((X - 2) - new_X) + abs(Y - new_Y)) > 2 and \
            current_state["board"][X - 1][Y] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif X - 2 >= 0 and current_state["board"][X - 2][Y] == 3 and (abs((X - 2) - new_X) + abs(Y - new_Y)) <= 2 and \
            current_state["board"][X - 1][Y] not in [1, 2]:
        reward += r_get_close_to_bomb
    if X - 3 >= 0 and current_state["board"][X - 3][Y] == 3 and (abs((X - 3) - new_X) + abs(Y - new_Y)) > 3 and \
            current_state["board"][X - 1][Y] not in [1, 2] and current_state["board"][X - 2][Y] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif X - 3 >= 0 and current_state["board"][X - 3][Y] == 3 and (abs((X - 3) - new_X) + abs(Y - new_Y)) <= 3 and \
            current_state["board"][X - 1][Y] not in [1, 2] and current_state["board"][X - 2][Y] not in [1, 2]:
        reward += r_get_close_to_bomb
    # 下
    if X + 1 <= 10 and current_state["board"][X + 1][Y] == 3 and (abs((X + 1) - new_X) + abs(Y - new_Y)) > 1:
        reward += r_get_away_from_bomb
    if X + 1 <= 10 and current_state["board"][X + 1][Y] == 3 and (abs((X + 1) - new_X) + abs(Y - new_Y)) == 1:
        reward += 2 * r_get_close_to_bomb
    if X + 2 <= 10 and current_state["board"][X + 2][Y] == 3 and (abs((X + 2) - new_X) + abs(Y - new_Y)) > 2 and \
            current_state["board"][X + 1][Y] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif X + 2 <= 10 and current_state["board"][X + 2][Y] == 3 and (abs((X + 2) - new_X) + abs(Y - new_Y)) <= 2 and \
            current_state["board"][X + 1][Y] not in [1, 2]:
        reward += r_get_close_to_bomb
    if X + 3 <= 10 and current_state["board"][X + 3][Y] == 3 and (abs((X + 3) - new_X) + abs(Y - new_Y)) > 3 and \
            current_state["board"][X + 1][Y] not in [1, 2] and current_state["board"][X + 2][Y] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif X + 3 <= 10 and current_state["board"][X + 3][Y] == 3 and (abs((X + 3) - new_X) + abs(Y - new_Y)) <= 3 and \
            current_state["board"][X + 1][Y] not in [1, 2] and current_state["board"][X + 2][Y] not in [1, 2]:
        reward += r_get_close_to_bomb

    # 左
    if Y - 1 >= 0 and current_state["board"][X][Y - 1] == 3 and (abs(X - new_X) + abs((Y - 1) - new_Y)) > 1:
        reward += r_get_away_from_bomb
    if Y - 1 >= 0 and current_state["board"][X][Y - 1] == 3 and (abs(X - new_X) + abs((Y - 1) - new_Y)) == 1:
        reward += 2 * r_get_close_to_bomb
    if Y - 2 >= 0 and current_state["board"][X][Y - 2] == 3 and (abs(X - new_X) + abs((Y - 2) - new_Y)) > 2 and \
            current_state["board"][X][Y - 1] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif Y - 2 >= 0 and current_state["board"][X][Y - 2] == 3 and (abs(X - new_X) + abs((Y - 2) - new_Y)) <= 2 and \
            current_state["board"][X][Y - 1] not in [1, 2]:
        reward += r_get_close_to_bomb
    if Y - 3 >= 0 and current_state["board"][X][Y - 3] == 3 and (abs(X - new_X) + abs((Y - 3) - new_Y)) > 3 and \
            current_state["board"][X][Y - 1] not in [1, 2] and current_state["board"][X][Y - 2] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif Y - 3 >= 0 and current_state["board"][X][Y - 3] == 3 and (abs(X - new_X) + abs((Y - 3) - new_Y)) <= 3 and \
            current_state["board"][X][Y - 1] not in [1, 2] and current_state["board"][X][Y - 2] not in [1, 2]:
        reward += r_get_close_to_bomb

    # 右
    if Y + 1 <= 10 and current_state["board"][X][Y + 1] == 3 and (abs(X - new_X) + abs((Y + 1) - new_Y)) > 1:
        reward += r_get_away_from_bomb
    if Y + 1 <= 10 and current_state["board"][X][Y + 1] == 3 and (abs(X - new_X) + abs((Y + 1) - new_Y)) == 1:
        reward += 2 * r_get_close_to_bomb
    if Y + 2 <= 10 and current_state["board"][X][Y + 2] == 3 and (abs(X - new_X) + abs((Y + 2) - new_Y)) > 2 and \
            current_state["board"][X][Y + 1] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif Y + 2 <= 10 and current_state["board"][X][Y + 2] == 3 and (abs(X - new_X) + abs((Y + 2) - new_Y)) <= 2 and \
            current_state["board"][X][Y + 1] not in [1, 2]:
        reward += r_get_close_to_bomb
    if Y + 3 <= 10 and current_state["board"][X][Y + 3] == 3 and (abs(X - new_X) + abs((Y + 3) - new_Y)) > 3 and \
            current_state["board"][X][Y + 1] not in [1, 2] and current_state["board"][X][Y + 2] not in [1, 2]:
        reward += r_get_away_from_bomb
    elif Y + 3 <= 10 and current_state["board"][X][Y + 3] == 3 and (abs(X - new_X) + abs((Y + 3) - new_Y)) <= 3 and \
            current_state["board"][X][Y + 1] not in [1, 2] and current_state["board"][X][Y + 2] not in [1, 2]:
        reward += r_get_close_to_bomb

    # 检查四角方向 左上
    if X - 1 >= 0 and Y - 1 >= 0 and current_state["board"][X - 1][Y - 1] == 3 and (abs((X - 1) - new_X)) + abs(
            (Y - 1) - new_Y) > 2:
        reward += r_get_away_from_bomb
    elif X - 1 >= 0 and Y - 1 >= 0 and current_state["board"][X - 1][Y - 1] == 3 and (abs((X - 1) - new_X)) + abs(
            (Y - 1) - new_Y) < 2:
        reward += r_get_close_to_bomb
    # 左下
    if X + 1 <= 10 and Y - 1 >= 0 and current_state["board"][X + 1][Y - 1] == 3 and (abs((X + 1) - new_X)) + abs(
            (Y - 1) - new_Y) > 2:
        reward += r_get_away_from_bomb
    elif X + 1 <= 10 and Y - 1 >= 0 and current_state["board"][X + 1][Y - 1] == 3 and (abs((X + 1) - new_X)) + abs(
            (Y - 1) - new_Y) < 2:
        reward += r_get_close_to_bomb
    # 右上
    if X - 1 >= 0 and Y + 1 <= 10 and current_state["board"][X - 1][Y + 1] == 3 and (abs((X - 1) - new_X)) + abs(
            (Y + 1) - new_Y) > 2:
        reward += r_get_away_from_bomb
    elif X - 1 >= 0 and Y + 1 <= 10 and current_state["board"][X - 1][Y + 1] == 3 and (abs((X - 1) - new_X)) + abs(
            (Y + 1) - new_Y) < 2:
        reward += r_get_close_to_bomb

    # 右下
    if X + 1 <= 10 and Y + 1 <= 10 and current_state["board"][X + 1][Y + 1] == 3 and (abs((X + 1) - new_X)) + abs(
            (Y + 1) - new_Y) > 2:
        reward += r_get_away_from_bomb
    elif X + 1 <= 10 and Y + 1 <= 10 and current_state["board"][X + 1][Y + 1] == 3 and (abs((X + 1) - new_X)) + abs(
            (Y - 1) + new_Y) < 2:
        reward += r_get_close_to_bomb

    return reward


def check_move_loop(action_list, reward, r_move_loop):
    check_list = [[1, 2, 1, 2],
                  [2, 1, 2, 1],
                  [3, 4, 3, 4],
                  [4, 3, 4, 3],
                  [1, 2, 3, 4],
                  [2, 1, 3, 4],
                  [3, 4, 1, 2],
                  [3, 4, 2, 1],
                  [1, 2, 4, 3],
                  [2, 1, 4, 3],
                  [4, 3, 1, 2],
                  [4, 3, 2, 1]]
    # if action_list[0] == action_list[2] and action_list[1] == action_list[3] and (
    #         action_list[0] and action_list[1] and action_list[2] and action_list[3] in [1, 2, 3, 4]) and \
    #         action_list[1] != action_list[0] and (action_list not in check_list):
    if action_list in check_list:
        reward += r_move_loop
    return reward


def check_dead_end(new_state, new_X, new_Y, action_list, reward, r_dead_end):
    if action_list[2] == 5 and action_list[3] == 1:
        if (new_Y - 1 < 0 or new_state["board"][new_X][new_Y - 1] in [1, 2, 3]) and \
                (new_Y + 1 > 10 or new_state["board"][new_X][new_Y + 1] in [1, 2, 3]) and \
                (new_X - 1 < 0 or new_state["board"][new_X - 1][new_Y] in [1, 2, 3]):
            reward += r_dead_end
            return reward
    elif action_list[2] == 5 and action_list[3] == 2:
        if (new_Y - 1 < 0 or new_state["board"][new_X][new_Y - 1] in [1, 2, 3]) and \
                (new_Y + 1 > 10 or new_state["board"][new_X][new_Y + 1] in [1, 2, 3]) and \
                (new_X + 1 > 10 or new_state["board"][new_X + 1][new_Y] in [1, 2, 3]):
            reward += r_dead_end
            return reward
    elif action_list[2] == 5 and action_list[3] == 3:
        if (new_X - 1 < 0 or new_state["board"][new_X - 1][new_Y] in [1, 2, 3]) and \
                (new_Y - 1 < 0 or new_state["board"][new_X][new_Y - 1] in [1, 2, 3]) and \
                (new_X + 1 > 10 or new_state["board"][new_X + 1][new_Y] in [1, 2, 3]):
            reward += r_dead_end
            return reward
    elif action_list[2] == 5 and action_list[3] == 4:
        if (new_X - 1 < 0 or new_state["board"][new_X - 1][new_Y] in [1, 2, 3]) and \
                (new_Y + 1 > 10 or new_state["board"][new_X][new_Y + 1] in [1, 2, 3]) and \
                (new_X + 1 > 10 or new_state["board"][new_X + 1][new_Y] in [1, 2, 3]):
            reward += r_dead_end
            return reward

    return reward


def check_avoid_flame(reward, r_avoid, current_grids):
    if 4 in current_grids and all((grid in [1, 2, 3, 4]) for grid in current_grids):
        reward += r_avoid
    return reward


def check_wood_and_power(current_state, new_X, new_Y, action_list, current_grids, reward, r_ignore_penalty):
    power = False
    # 先检查power 更重要
    if ((6 or 7 or 8) in current_grids) and current_state["board"][new_X][new_Y] not in [6, 7, 8] and (
            5 not in action_list) and \
            current_state["ammo"] != 0:
        reward += r_ignore_penalty
        power = True
    # if power is False and (2 in current_grids) and (5 not in action_list) and current_state["ammo"] != 0:
    #     reward += r_ignore_penalty

    return reward


def featurize2D(states):
    partially_obs = True
    # 共18个矩阵
    X = states["position"][0]
    Y = states["position"][1]
    shape = (11, 11)

    # path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4
    def get_partially_obs(states):
        board = np.zeros(shape)
        for x in range(10):
            for y in range(10):
                if X - 4 <= x <= X + 4 and Y - 4 <= y <= Y + 4:
                    board[x][y] = states["board"][x][y]
        states["board"] = board
        return states

    def get_matrix(board, key):
        res = board[key]
        return res.reshape(shape).astype(np.float64)

    def get_map(board, item):
        map = np.zeros(shape)
        map[board == item] = 1
        return map

    if partially_obs:
        states = get_partially_obs(states)

    board = get_matrix(states, "board")

    path = get_map(board, 0)
    rigid = get_map(board, 1)
    wood = get_map(board, 2)
    bomb = get_map(board, 3)
    flame = get_map(board, 4)
    fog = get_map(board, 5)
    agent1 = get_map(board, 10)
    agent2 = get_map(board, 11)
    agent3 = get_map(board, 12)
    agent4 = get_map(board, 13)

    power_up = []
    for row in board:
        new_row = []
        for num in row:
            if num == 6 or num == 7 or num == 8:
                new_row.append(1)
            else:
                new_row.append(0.0)
        power_up.append(new_row)

    bomb_blast_strength = get_matrix(states, 'bomb_blast_strength')
    bomb_life = get_matrix(states, 'bomb_life')
    bomb_moving_direction = get_matrix(states, 'bomb_moving_direction')
    flame_life = get_matrix(states, 'flame_life')

    ammo_2D, blast_strength_2D, can_kick_2D = rebuild_1D_element(states)

    feature2D = [path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4, bomb_blast_strength,
                 bomb_life, bomb_moving_direction, flame_life, ammo_2D, blast_strength_2D, can_kick_2D]

    return np.array(feature2D)


def rebuild_1D_element(states):
    ammo = states["ammo"]
    ammo_2D = np.full((11, 11), ammo)

    blast_strength = states["blast_strength"]
    blast_strength_2D = np.full((11, 11), blast_strength)

    can_kick = states["can_kick"]
    can_kick_2D = np.full((11, 11), int(can_kick))

    return ammo_2D, blast_strength_2D, can_kick_2D