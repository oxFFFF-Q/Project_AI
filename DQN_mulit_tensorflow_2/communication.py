import numpy as np


def message(current_state):
    """
    FFA: message = the maximum distance bomb coordinate with the agent
    Radio: message = bomb coordinates 4-7 away from teammates,
                     Or the coordinates of the bomb with the maximum distance from teammates,
                     Or the maximum distance bomb coordinate with the agent
    input: current_state
    output: message, current_state processed using the received message
    """

    position_agent = current_state["position"]
    board = current_state["board"]
    teammate = current_state["teammate"].value
    list_position_bomb = np.argwhere(board == 3).tolist()
    position_teammate = np.argwhere(board == teammate).tolist()
    r_mess = current_state['message']

    # Correct the inverted coordinates and Ignore bombs on the edge
    if len(list_position_bomb) != 0:
        for index, position_bomb in enumerate(list_position_bomb):
            position_bomb_ = [position_bomb[1], position_bomb[0]]
            list_position_bomb[index] = position_bomb_
            if position_bomb[0] < 1 or position_bomb[0] > 9 or \
                    position_bomb[1] < 1 or position_bomb[1] > 9:
                list_position_bomb.pop(index)

    # Add the received bomb position to the current state
    if r_mess != (0, 0):
        current_state["board"][r_mess[0]+1, r_mess[1]+1] = 3

    if teammate == 9 or len(position_teammate) == 0:
        if len(list_position_bomb) != 0:
            return position_max_dis(position_agent, list_position_bomb), current_state
        else:
            return [0, 0], current_state
    else:
        if len(list_position_bomb) != 0:
            return position_max_dis_limit(position_teammate[0], list_position_bomb), current_state
        else:
            return [0, 0], current_state


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def position_max_dis(x, list_y):
    list_distance = []
    for y in list_y:
        list_distance.append(distance(x, y))
    return [list_y[np.argmax(np.array(list_distance))][0] - 1, list_y[np.argmax(np.array(list_distance))][1] - 1]


def position_max_dis_limit(x, list_y):
    limit = [4, 7]
    list_distance = []
    for y in list_y:
        dis = distance(x, y)
        if limit[0] < dis < limit[1]:
            list_distance.append(dis)
    if len(list_distance) != 0:
        return [list_y[np.argmax(np.array(list_distance))][0] - 1, list_y[np.argmax(np.array(list_distance))][1] - 1]
    else:
        return [0, 0]
