import numpy as np
import constants
from pommerman import constants


def reward_shaping(current_state, new_state, action, result):
    r_win = 1
    r_lose = -1

    r_wood = 0.03
    r_powerup = 0.05
    # r_put_bomb = 0.3
    r_put_bomb = 0
    r_put_bomb_near_enemy = 0.05
    r_kick = 0.02

    r_get_away_from_bomb = 0.003
    r_move = 0.004
    r_stay = -0.002

    reward = 0

    X = current_state["position"][0]
    Y = current_state["position"][1]
    new_X = new_state["position"][0]
    new_Y = new_state["position"][1]

    # 达到最大步数，为平局
    # if steps == constants.MAX_STEPS + 1:
    #     return reward

    if result == 1:
        reward += r_win

    # 检查是否踢了炸弹
    if current_state["can_kick"] is True and new_state["can_kick"] is False:
        reward += r_kick

    if action == 0:
        reward += r_stay
        reward = check_in_flame(current_state, new_state, reward, r_lose)
        return reward

    # 移动reward
    if action == 1:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_stay
        else:
            reward += r_move
        # 检查是否吃到道具
        if Y - 1 >= 0:
            if current_state["board"][X][Y - 1] == 6 or current_state["board"][X][Y - 1] == 7 or \
                    current_state["board"][X][Y - 1] == 8:
                reward += r_powerup
        reward = check_in_flame(current_state, new_state, reward, r_lose)
        #reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb)
        return reward

    if action == 2:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_stay
        else:
            reward += r_move
        # 检查是否吃到道具
        if Y + 1 <= 10:
            if current_state["board"][X][Y + 1] == 6 or current_state["board"][X][Y + 1] == 7 or \
                    current_state["board"][X][Y + 1] == 8:
                reward += r_powerup
        reward = check_in_flame(current_state, new_state, reward, r_lose)
        #reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb)
        return reward

    if action == 3:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_stay
        else:
            reward += r_move
        # 检查是否吃到道具
        if X - 1 >= 0:
            if current_state["board"][X - 1][Y] == 6 or current_state["board"][X - 1][Y] == 7 or \
                    current_state["board"][X - 1][Y] == 8:
                reward += r_powerup
        reward = check_in_flame(current_state, new_state, reward, r_lose)
        #reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb)
        return reward

    if action == 4:
        # 检查是否撞墙
        if current_state["position"] == new_state["position"]:
            reward += r_stay
        else:
            reward += r_move
        # 检查是否吃到道具
        if X + 1 <= 10:
            if current_state["board"][X + 1][Y] == 6 or current_state["board"][X + 1][Y] == 7 or \
                    current_state["board"][X + 1][Y] == 8:
                reward += r_powerup
        reward = check_in_flame(current_state, new_state, reward, r_lose)
        #reward = check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb)
        return reward

    # 放炸弹reward，包括检查wood， 敌人等
    if action == 5:

        blast_strength = current_state["blast_strength"]
        reward = check_in_flame(current_state, new_state, reward, r_lose)

        if current_state["ammo"] != 0:
            reward += r_put_bomb
            # 判断炸弹上方是否有墙
            for strength in range(1, blast_strength + 1):
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
                elif current_state["board"][X][Y - strength] == 11 or current_state["board"][X][Y - strength] == 12 or \
                        current_state["board"][X][Y - strength] == 13:
                    reward += r_put_bomb_near_enemy

            # 判断炸弹下方是否有墙
            for strength in range(1, blast_strength + 1):
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
                elif current_state["board"][X][Y + strength] == 11 or current_state["board"][X][Y + strength] == 12 or \
                        current_state["board"][X][Y + strength] == 13:
                    reward += r_put_bomb_near_enemy

            # 判断炸弹左方是否有墙
            for strength in range(1, blast_strength + 1):
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
                elif current_state["board"][X - strength][Y] == 11 or current_state["board"][X - strength][Y] == 12 or \
                        current_state["board"][X - strength][Y] == 13:
                    reward += r_put_bomb_near_enemy

            # 判断炸弹右方是否有墙
            for strength in range(1, blast_strength + 1):
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
                elif current_state["board"][X + strength][Y] == 11 or current_state["board"][X + strength][Y] == 12 or \
                        current_state["board"][X + strength][Y] == 13:
                    reward += r_put_bomb_near_enemy

        elif current_state["ammo"] == 0:
            reward += r_stay

        return reward

        # 未移动位置reward

    return reward


def check_in_flame(current_state, new_state, reward, r_in_flame):
    # 若agent与火焰位置重叠，则死亡，返回reward
    X = current_state["position"][0]
    Y = current_state["position"][1]
    new_X = new_state["position"][0]
    new_Y = new_state["position"][1]

    if current_state["flame_life"][X][Y] == 0 and new_state["flame_life"][new_X][new_Y] != 0:
        reward += r_in_flame
    return reward


def check_and_away_from_bomb(current_state, X, Y, new_X, new_Y, reward, r_get_away_from_bomb):
    bomb = np.argwhere(current_state["board"] == 3)
    if len(bomb) > 0:
        for i in range(len(bomb)):
            distance_current = abs(X - bomb[i][0]) + abs(Y - bomb[i][1])
            distance_new = abs(new_X - bomb[i][0]) + abs(new_Y - bomb[i][1])
            if distance_current <= 2 and (distance_new - distance_current) > 0:
                reward += r_get_away_from_bomb
    return reward


def featurize2D(states):
    feature2D = []
    # 共18个矩阵
    for board in rebuild_board(states["board"]):
        feature2D.append(board)

    feature2D.append(states["bomb_blast_strength"].tolist())
    feature2D.append(states["bomb_life"].tolist())
    feature2D.append(states["bomb_moving_direction"].tolist())
    feature2D.append(states["flame_life"].tolist())

    ammo_2D, blast_strength_2D, can_kick_2D = rebuild_1D_element(states)

    feature2D.append(ammo_2D)
    feature2D.append(blast_strength_2D)
    feature2D.append(can_kick_2D)

    return np.array(feature2D)


def rebuild_board(board):
    # 将board中数据分离，2D化
    path = []
    for row in board:
        new_row = []
        for num in row:
            if num == 0:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        path.append(new_row)

    rigid = []
    for row in board:
        new_row = []
        for num in row:
            if num == 1:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        rigid.append(new_row)

    wood = []
    for row in board:
        new_row = []
        for num in row:
            if num == 2:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        wood.append(new_row)

    bomb = []
    for row in board:
        new_row = []
        for num in row:
            if num == 3:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        bomb.append(new_row)

    flame = []
    for row in board:
        new_row = []
        for num in row:
            if num == 4:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        flame.append(new_row)
    """
    暂时用不到fog
    fog =[]
    for row in board:
        new_row = []
        for num in row:
            if num == 4:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        fog.append(new_row)
    """

    fog = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ]

    power_up = []
    for row in board:
        new_row = []
        for num in row:
            if num == 6 or num == 7 or num == 8:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        power_up.append(new_row)

    agent1 = []
    # 如果是9为此处为agent,则取1.0
    for row in board:
        new_row = []
        for num in row:
            if num == 9:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        agent1.append(new_row)

    agent2 = []
    # 如果是10为此处为agent,则取1.0
    for row in board:
        new_row = []
        for num in row:
            if num == 10:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        agent2.append(new_row)

    agent3 = []
    # 如果是11为此处为agent,则取1.0
    for row in board:
        new_row = []
        for num in row:
            if num == 11:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        agent3.append(new_row)

    agent4 = []
    # 如果是12为此处为agent,则取1.0
    for row in board:
        new_row = []
        for num in row:
            if num == 12:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        agent4.append(new_row)

    return path, rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4


def rebuild_1D_element(states):
    ammo = states["ammo"]
    ammo_2D = np.full((11, 11), ammo)

    blast_strength = states["blast_strength"]
    blast_strength_2D = np.full((11, 11), blast_strength)

    can_kick = states["can_kick"]
    can_kick_2D = np.full((11, 11), int(can_kick))

    return ammo_2D, blast_strength_2D, can_kick_2D
