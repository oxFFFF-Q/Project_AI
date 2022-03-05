# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def plot(name_png, title_graph, list_xy, list_label):
    """
    将同一路径下的csv文件夹内所有文件，转换为一张折线图，并保存到同意路径下的graphs文件夹中
    input:
        name_png: 生成图片名称
        title_graph: 图表题目
        List_label:  图表内折线标签，Format：['label1', 'label2', ....],数量与csv文件夹内文件个数一致
    """
    list_data = load_csv()
    # list_maker = [',', '.', 'o', 'v', '^', '<', '>', '+', 'x', 'p', '*']
    list_maker = [',', ',', ',', ',', ',', ',', ',', '+', 'x', 'p', '*']
    if len(list_data) != 0:
        s = len(list_data[0])
        x = [i + 1 for i in range(s)]
        plt.figure(1)
        plt.title(title_graph)
        plt.xlabel(list_xy[0])
        plt.ylabel(list_xy[1])
        for index, list_d in enumerate(list_data):
            plt.plot(x, list_d, label=list_label[index], marker=list_maker[index],
                     alpha=0.8, linewidth=0.5)

    # plt.gca().yaxis.set_major_locator(MultipleLocator(int(len(List_data[0]) / 10)))
    if len(list_label) > 1:
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
    png_path = get_path(name_png, 'graphs')
    plt.savefig(png_path)
    plt.show()


def load_csv():
    # 默认载入csv文件夹内所有文件
    check_folder(['graphs'])
    path_folder = os.path.join(os.getcwd().replace('\\', '/'), 'csv')
    dirs = os.listdir(path_folder)
    list_data = []
    if len(dirs) != 0:
        for d in dirs:
            path_csv = os.path.join(path_folder, d)
            file = open(path_csv)  # 打开csv文件
            reader = csv.reader(file)  # 读取csv文件
            data = list(reader)  # csv数据转换为列表
            data = np.array(data)
            data = data.astype(float).tolist()
            list_data.append([r[col] for r in data for col in range(len(data[0]))])
    return list_data


def get_path(name, name_floder):
    folder_path = os.getcwd().replace('\\', '/')
    png_path = os.path.join(folder_path, name_floder, name)
    return png_path


def check_folder(list_name):
    current_path = os.getcwd().replace('\\', '/')
    if len(list_name) != 0:
        for name in list_name:
            path_folder = os.path.join(current_path, name)
            if not os.path.exists(path_folder):
                os.mkdir(path_folder)


if __name__ == '__main__':

    # plot('name_png', 'title_graph', ['episodes', 'result'], ['result', 'reward'])
    # plot('Reward_rainbow', 'Rainbow DQN', ['episodes', 'reward'],
    #      ['DQN', 'DQN with data augmentation', 'Double DQN', 'Dueling DQN',
    #       'DQN + noisy network', 'DQN + priority memory', 'double DQN + priority'])
    plot('Reward_DQN + priority memory', 'DQN + priority memory', ['episodes', 'reward'], ['Basic DQN'])
