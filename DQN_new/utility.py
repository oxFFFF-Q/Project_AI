import numpy as np


def featurize(env, states):
    '''
    Converts the states(dict) into list of 1D numpy arrays

    Input:
    - env: gym environment
    - states: list[num_agents, dict(15)] for each agent
    Output:
    - feature: list[num_agents, 372]
    '''
    feature = []
    for state in states:
        feature.append(env.featurize(state).tolist())
        # changes to 1D numpy array
    return feature


def featurize2D(states):
    feature2D = []
    # 共9个矩阵
    for board in rebuild_board(states[0]["board"]):
        feature2D.append(board)

    feature2D.append(states[0]["bomb_blast_strength"].tolist())
    feature2D.append(states[0]["bomb_life"].tolist())
    feature2D.append(states[0]["bomb_moving_direction"].tolist())
    feature2D.append(states[0]["flame_life"].tolist())

    return feature2D


def rebuild_board(board):
    # 将board中数据分离，2D化
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
    power_up = []
    for row in board:
        new_row = []
        for num in row:
            if num == 6 or num == 7 or num == 8:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        power_up.append(new_row)

    agents = []
    # 如果是9,10,11,12代为此处为agent,则取1.0
    for row in board:
        new_row = []
        for num in row:
            if num == 9 or num == 10 or num == 11 or num == 12:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        agents.append(new_row)

    return rigid, wood,bomb, power_up, agents
