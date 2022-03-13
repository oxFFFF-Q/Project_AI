import random

import numpy as np


def action_filter(current_state, action_list):
    """Filtrate the action to improve survival rate"""
    X = current_state["position"][0]
    Y = current_state["position"][1]
    # Record the movable positions around the agent
    move_list = make_move_list(current_state, X, Y)
    # If agent cannot move, then excute action 0 (waiting to die)
    if move_list is None:
        return 0
    # Get the moveable position from agent after statistical analysing various actions
    moveable_position = make_move_position(move_list, X, Y, action_list)
    # Statistical analysing all bombs in visual range
    bomb_list = make_bomb_list(current_state)
    # Grade the danger level of the new location
    moveable_position_score = make_dangerous_list(current_state, moveable_position, bomb_list, action_list)

    actions = []
    # Use a safe location if one exists
    for action in moveable_position_score:
        if action[3] == 0:
            actions.append(action[2])
    # If a original action is safe, directly choose the original action
    if action_list[-1] in actions:
        return action_list[-1]

    # If there is no safe position, choose the position of risk degree 1
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 1:
                actions.append(action[2])
    # If a original action is safe, directly choose the original action
    if action_list[-1] in actions:
        return action_list[-1]

    # If there is no position of risk degree 1, directly choose the position of risk degree 2
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 2:
                actions.append(action[2])

    # If there is no position of risk degree 2, choose the position of risk degree 3
    if actions is None:
        actions = []
        for action in moveable_position_score:
            if action[3] == 3:
                actions.append(action[2])

    if actions is None:
        return random.randint(0, 5)

    # If multiple actions are available, the non-0 action is preferred
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
    """Check if the action is executable (meaningful)"""

    def check_bomb_action(current_state):
        """Check if Action 5 makes sense"""
        meaningful = False
        X = current_state["position"][0]
        Y = current_state["position"][1]
        blast_strength = current_state["blast_strength"]
        # Determine if there's a wall to the left of the bomb
        for strength in range(1, blast_strength):
            # Check if an agent is outside the map boundaries
            if Y - strength < 0:
                break
            # If it is rigid, break
            elif current_state["board"][X][Y - strength] == 1:
                break
            # Check if there is wood, then set meaningful to true
            elif current_state["board"][X][Y - strength] == 2:
                meaningful = True
                return meaningful
            # It makes sense if there's an enemy in the blast range
            elif current_state["board"][X][Y - strength] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # Determine if there's a wall to the right of the bomb
        for strength in range(1, blast_strength):
            # Check if an agent is outside the map boundaries
            if Y + strength > 10:
                break
                # If it is rigid, break
            elif current_state["board"][X][Y + strength] == 1:
                break
            # Check if there is wood, then set meaningful to true
            elif current_state["board"][X][Y + strength] == 2:
                meaningful = True
                return meaningful
            # It makes sense if there's an enemy in the blast range
            elif current_state["board"][X][Y + strength] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # Determine if there is a wall above the bomb
        for strength in range(1, blast_strength):
            # Check if an agent is outside the map boundaries
            if X - strength < 0:
                break
            # If it is rigid, break
            elif current_state["board"][X - strength][Y] == 1:
                break
            # Check if there is wood, then set meaningful to true
            elif current_state["board"][X - strength][Y] == 2:
                meaningful = True
                return meaningful
            # It makes sense if there's an enemy in the blast range
            elif current_state["board"][X - strength][Y] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful

        # Determine if there is a wall under the bomb
        for strength in range(1, blast_strength):
            # Check if an agent is outside the map boundaries
            if X + strength > 10:
                break
            # If it is rigid, break
            elif current_state["board"][X + strength][Y] == 1:
                break
            # Check if there is wood, then set meaningful to true
            elif current_state["board"][X + strength][Y] == 2:
                meaningful = True
                return meaningful
            # It makes sense if there's an enemy in the blast range
            elif current_state["board"][X + strength][Y] in [10, 11, 12, 13]:
                meaningful = True
                return meaningful
        return meaningful

    def check_moveable(current_state, X, Y):
        """Check whether an agent can be moved in this position"""
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
    """Calculate risk degree of the movable position"""
    # First two columns, post movement position, third column, movement, fourth column, risk degree
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
        # Check for obstacles between mines and agents
        block = False
        if position_agent[0] != position_bomb[0]:
            for index in range(1, abs(position_agent[0] - position_bomb[0])):
                # Check for wood and rigid
                if current_state["board"][min(position_agent[0], position_bomb[0]) + index][position_agent[1]] in [1,
                                                                                                                   2]:
                    block = True
                    break
        elif position_agent[1] != position_bomb[1]:
            for index in range(1, abs(position_agent[1] - position_bomb[1])):
                # Check for wood and rigid
                if current_state["board"][position_agent[0]][min(position_agent[1], position_bomb[1]) + index] in [1,
                                                                                                                   2]:
                    block = True
                    break

        return block

    # Check dead ends
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

    # Check the flame
    def check_flame(current_state, moveable_position):
        for position in moveable_position:
            if current_state["flame_life"][position[0]][position[1]] != 0:
                position[-1] = 3
        return moveable_position

    # bomb_location_all = np.where(np.array(current_state["board"]) == 3)
    # if len(bomb_location_all[0]) == 0:
    #     return moveable_position
    moveable_position = check_dead_end(current_state, moveable_position, action_list)

    # Check for flames, even if there is no bomb in sight
    if len(bomb_list) == 0:
        moveable_position = check_flame(current_state, moveable_position)
        return moveable_position
        '''
            Moveable position is classified according to the danger level.
            when agent is threatened by multiple bombs or flames in the same location,
            the highest danger level is recorded first.
        '''
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
