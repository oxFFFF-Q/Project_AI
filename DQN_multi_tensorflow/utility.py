import numpy as np
from pommerman import utility


def featurize(states):
    '''
    Converts the states(dict) into list of 1D numpy arrays

    Input:
    - env: gym environment
    - states: list[num_agents, dict(15)] for each agent
    Output:
    - feature: list[num_agents, 372]
    '''

    # length = len(env.featurize(states[0]).tolist())
    # list = env.featurize(states[0]).tolist()
    # states = states[0]
    local = featurize2D(states)
    """
    board = states["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = states["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = states["bomb_life"].reshape(-1).astype(np.float32)
    bomb_moving_direction = states["bomb_moving_direction"].reshape(-1).astype(np.float32)
    flame_life = states["flame_life"].reshape(-1).astype(np.float32)
    local.append(board.tolist())
    local.append(bomb_blast_strength.tolist())
    local.append(bomb_life.tolist())
    local.append(bomb_moving_direction.tolist())
    local.append(flame_life.tolist())
    """
    feature = {'local': local}
    additional = []
    position = utility.make_np_float(states["position"])
    ammo = utility.make_np_float([states["ammo"]])  # fff
    blast_strength = utility.make_np_float([states["blast_strength"]])
    can_kick = utility.make_np_float([states["can_kick"]])
    teammate = utility.make_np_float([states["teammate"].value])
    enemies = utility.make_np_float([e.value for e in states["enemies"]])
    # print(position, ammo, blast_strength, can_kick, teammate, enemies)
    """
    additional.append(position.tolist())
    additional.append(ammo.tolist())
    additional.append(blast_strength.tolist())
    additional.append(can_kick.tolist())
    additional.append(teammate.tolist())
    additional.append(enemies.tolist())
    """
    # print(additional)
    # position占两个数，所以你要取ammo的话就要取additional[2]
    additional = np.concatenate(
        (position, ammo,
         blast_strength, can_kick, teammate, enemies))

    feature['additional'] = additional.tolist()
    return feature


def reward_shaping(minibatch):
    mini_batch = minibatch

    statesl, statesa, actions, rewards, next_statesl, next_statesa, done = [], [], [], [], [], [], []
    # epistep = []

    for transition in mini_batch:  # transition: tuple
        state, additions, re, new_state, done_mask = transition
        statesl.append(state['local'])
        statesa.append(state['additional'])
        actions.append([additions])
        rewards.append([re])
        next_statesl.append(new_state['local'])
        next_statesa.append(new_state['additional'])
        done.append([done_mask])
    """
    sl_lst = np.array(sl_lst)
    sa_lst = np.array(sa_lst)
    a_lst = np.array(a_lst)
    r_lst = np.array(r_lst)
    sl_prime_list = np.array(sl_prime_list)
    sa_prime_list = np.array(sa_prime_list)
    done_mask_list = np.array(done_mask_list)
    """
    computed_reward = []
    for local, additions, action, state_local, state_additions, re in zip(next_statesl, next_statesa, actions, statesl,
                                                                          statesa, rewards):
        computed_reward.append(reward(local, additions, action, state_local, state_additions, re))
    # 这是得到的reward

    return computed_reward


def featurize2D(states):
    feature2D = []
    # 共9个矩阵
    for element_board in rebuild_board(states["board"]):
        feature2D.append(element_board)

    feature2D.append(states["bomb_blast_strength"].tolist())
    feature2D.append(states["bomb_life"].tolist())
    feature2D.append(states["bomb_moving_direction"].tolist())
    feature2D.append(states["flame_life"].tolist())

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

    # fog部分
    fog = []
    for row in board:
        new_row = []
        for num in row:
            if num == 5:
                new_row.append(1.0)
            else:
                new_row.append(0.0)
        fog.append(new_row)

    # 充数的fog
    # fog = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

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

    return rigid, wood, bomb, flame, fog, power_up, agent1, agent2, agent3, agent4


def reward(state_local, state_additions, action, new_state_local, new_state_additions, rewards):
    # set up reward
    r_wood = 0.1
    r_powerup = 0.3
    r_put_bomb = 0.08
    r_win = 1
    r_fail = -5
    r_kick = 0.3
    r_kill_enemy_maybe = 0.5
    r_dies = -3

    rigid = np.array(state_local[0])
    wood = np.array(state_local[1])
    bomb = np.array(state_local[2])
    agents = np.array(state_local[5])
    # power_up = featurel[3]
    position0 = int(state_additions[0])
    position1 = int(state_additions[1])
    p0 = int(new_state_additions[0])
    p1 = int(new_state_additions[1])
    ammo = int(state_additions[2])
    blast_strength = int(state_additions[3])
    can_kick = int(state_additions[4])
    teammate = int(state_additions[5])
    enemie1 = int(state_additions[6])
    enemie2 = int(state_additions[7])
    rewards = np.array(rewards)
    reward = 0
    # sagents = sl[4]
    sbomb = np.array(new_state_local[2])

    # reward_done
    # print(rewards)
    if rewards == 1:
        reward += r_win
    if rewards == -1:
        reward += r_fail

    # reward_powerup
    sammo = int(new_state_additions[2])
    if ammo > 1 and ammo > sammo:
        reward += r_powerup
    sstrength = int(new_state_additions[3])
    if blast_strength > sstrength:
        reward += r_powerup
    skick = int(new_state_additions[4])
    if can_kick and not skick:
        reward += r_powerup
    # print(action)

    # reward_wood
    if action == 5:
        reward += r_put_bomb
        bomb_flame = build_flame(position0, position1, rigid, blast_strength)
        num_wood = np.count_nonzero(wood * bomb_flame == 1)
        reward += num_wood * r_wood
        '''
        # test
        print('rigid')
        print(rigid)
        print('position_bomb')
        print(position_bomb)
        print('f')
        print(f)
        print('l')
        print(l)
        print('bomb_flame')
        print(bomb_flame)
        print('num_wood')
        print(num_wood)
        print('-------------------------------------')
        '''
    """
    exist_bomb = []
    for row, rowbomb in enumerate(bomb):
        for col, _ in enumerate(rowbomb):
            if bomb[row, col] == 1:
                exist_bomb.append((row, col))
    #print(bomb)
    #print(exist_bomb)

    if exist_bomb:
        for ebomb in exist_bomb:
            bomb_flame1 = self.build_flame(ebomb[0], ebomb[1], rigid, blast_strength)
            if bomb_flame1[position0, position1] == 1:
                reward -= 0.5
            #print(bomb_flame1)
    """
    # reward_kick
    if sbomb[position0, position1] == 1 and rewards != -1:
        reward += r_kick
    '''
    # reward_kill_enemy
    enemy_position = []              #需要知道敌人位置
    if int(action.item()) == 5:
        bomb_position = np.array([position0,position1])
        bomb_flame = self.build_flame(position0, position1, rigid, blast_strength)
    if bomb_position in np.argwhere(bomb==1) and np.argwhere(enemy_position*bomb_flame == 1).size != 0:
            reward += r_kill_enemy_maybe
    '''

    '''
    # reward_dies
    if is_alive == 0:
        reward += r_dies
    '''

    return reward


def build_flame(position0, position1, rigid, blast_strength):
    position_bomb = np.array([position0, position1])
    m = position_bomb[0]
    n = position_bomb[1]
    l = blast_strength - 1
    f = [l, l, l, l]  # Scope of flame: up down left right
    bomb_flame = np.zeros_like(rigid)

    # 判断实体墙或边界是否阻断火焰
    flame_up = np.zeros_like(bomb_flame)
    flame_down = np.zeros_like(bomb_flame)
    flame_left = np.zeros_like(bomb_flame)
    flame_right = np.zeros_like(bomb_flame)
    if m - f[0] < 0:  # 上边界
        f[0] = m
    flame_up[m - f[0]:m, n] = 1
    if m + f[1] > bomb_flame.shape[0] - 1:  # 下边界
        f[1] = bomb_flame.shape[0] - 1 - m
    flame_down[m + 1:m + f[1] + 1, n] = 1
    if n - f[2] < 0:  # 左边界
        f[2] = n
    flame_left[m, n - f[2]:n] = 1
    if n + f[3] > bomb_flame.shape[0] - 1:  # 右边界
        f[3] = bomb_flame.shape[0] - 1 - n
    flame_right[m, n + 1:n + f[3] + 1] = 1

    rigid_0 = flame_up * rigid
    rigid_1 = flame_down * rigid
    rigid_2 = flame_left * rigid
    rigid_3 = flame_right * rigid
    if np.argwhere(rigid_0 == 1).size != 0:  # 上实体墙
        rigid_up = np.max(np.argwhere(rigid_0 == 1)[:, 0][0])
        if rigid_up >= m - f[0]:
            f[0] = m - rigid_up - 1
    if np.argwhere(rigid_1 == 1).size != 0:  # 下实体墙
        rigid_down = np.min(np.argwhere(rigid_1 == 1)[:, 0][0])
        if rigid_down <= m + f[1]:
            f[1] = rigid_down - m - 1
    if np.argwhere(rigid_2 == 1).size != 0:  # 左实体墙
        rigid_left = np.max(np.argwhere(rigid_2 == 1)[0, :][1])
        if rigid_left >= n - f[2]:
            f[2] = n - rigid_left - 1
    if np.argwhere(rigid_3 == 1).size != 0:  # 右实体墙
        rigid_right = np.min(np.argwhere(rigid_3 == 1)[0, :][1])
        if rigid_right <= n + f[3]:
            f[3] = rigid_right - n - 1
    bomb_flame[m - f[0]:m + f[1] + 1, n] = 1
    bomb_flame[m, n - f[2]:n + f[3] + 1] = 1

    '''
    # test
    print('rigid')
    print(rigid)
    print('position_bomb')
    print(position_bomb)
    print('f')
    print(f)
    print('l')
    print(l)
    print('bomb_flame')
    '''
    # print(bomb_flame)
    # print(blast_strength)
    '''
    print('num_wood')
    print(num_wood)
    print('-------------------------------------')
    '''
    return bomb_flame
