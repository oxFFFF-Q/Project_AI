import numpy as np

def map_filter(board):
    f1 = np.zeros((11, 11))
    f1[0:3, 0:6] = 1
    f3 = np.zeros((11, 11))
    f3[3:6, 0:6] = 1
    f5 = np.zeros((11, 11))
    f5[6:9, 0:6] = 1
    f7 = np.zeros((11, 11))
    f7[9:11, 0:6] = 1

    f2 = np.zeros((11, 11))
    f2[0:3, 6:11] = 1
    f4 = np.zeros((11, 11))
    f4[3:6, 6:11] = 1
    f6 = np.zeros((11, 11))
    f6[6:9, 6:11] = 1
    f8 = np.zeros((11, 11))
    f8[9:11, 6:11] = 1

    n1 = np.argwhere(board * f1 > 0).shape[0]
    n2 = np.argwhere(board * f2 > 0).shape[0]
    n3 = np.argwhere(board * f3 > 0).shape[0]
    n4 = np.argwhere(board * f4 > 0).shape[0]
    n5 = np.argwhere(board * f5 > 0).shape[0]
    n6 = np.argwhere(board * f6 > 0).shape[0]
    n7 = np.argwhere(board * f7 > 0).shape[0]
    n8 = np.argwhere(board * f8 > 0).shape[0]

    arr_num = np.array([n1, n2, n3, n4, n5, n6, n7, n8])
    max_num1 = np.argmax(arr_num)
    arr_num[max_num1] = np.min(arr_num)
    max_num2 = np.argmax(arr_num)
    if max_num1 == max_num2:
        max_num1 = 0
        max_num2 = 0

    return max_num1, max_num2


def add_message(action, state_feature):
    bomb = state_feature[3]

    # 队友观测到炸弹最多的两个区域
    m1, m2 = map_filter(bomb)

    message = (action, m1, m2)
    return message
