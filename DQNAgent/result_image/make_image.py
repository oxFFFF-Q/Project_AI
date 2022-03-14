# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def plot(name_png, title_graph, list_xy, list_label, range_epi):
    """
    make a line chart by all the csv from data and save it
    input:
        name_png: name of the diagram
        title_graph: title of the diagram
        List_label:  labels of the line ，Format：['label1', 'label2', ....]
    """
    list_data = load_csv()
    # list_maker = [',', '.', 'o', 'v', '^', '<', '>', '+', 'x', 'p', '*']
    list_maker = [',', ',', ',', ',', ',', ',', ',', ',', ',', 'p', '*']
    if len(list_data) != 0:
        plt.figure(1)
        plt.title(title_graph)
        plt.xlabel(list_xy[0])
        plt.ylabel(list_xy[1])
        for index, list_d in enumerate(list_data):
            s = len(list_d)
            x = [i + 1 for i in range(s)]
            x, y = epi_display_range(x, list_d, range_epi)
            plt.plot(x, y, label=list_label[index], marker=list_maker[index],
                     alpha=0.9, linewidth=1)

    # plt.gca().yaxis.set_major_locator(MultipleLocator(int(len(List_data[0]) / 10)))
    if len(list_label) > 1:
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
    png_path = get_path(name_png, 'graphs')
    plt.savefig(png_path)
    plt.show()


def load_csv():
    # load all the csv
    check_folder(['graphs'])
    path_folder = os.path.join(os.getcwd().replace('\\', '/'), 'csv')
    dirs = os.listdir(path_folder)
    dirs.sort(key=lambda x: int(x[:-11]))
    list_data = []
    if len(dirs) != 0:
        for d in dirs:
            path_csv = os.path.join(path_folder, d)
            file = open(path_csv)
            reader = csv.reader(file)
            data = list(reader)
            data = np.array(data)
            data = data.astype(float).tolist()
            average_data = sum_average(data)
            # list_data.append([r[col] for r in data for col in range(len(data[0]))])
            list_data.append(average_data)
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


def sum_average(list_data):
    list_data_ave = []
    sum_data = 0
    average = []
    for index, data in enumerate(list_data):
        sum_data += data[0]
        average.append(sum_data / (index + 1))
    return average


def epi_display_range(x, y, range_epi):
    return x[range_epi[0]:range_epi[1]], y[range_epi[0]:range_epi[1]]


if __name__ == '__main__':
    # plot('name_png', 'title_graph', ['episodes', 'result'], ['result', 'reward'], [100, 800])
    plot('Reward_rainbow', 'Rainbow DQN', ['Episodes', 'Average reward'],
         ['DQN', 'DQN with data augmentation', 'Double DQN', 'Dueling DQN',
          'DQN + noisy network', 'DQN + priority memory', 'Multi-step DQN',
          'Double DQN + priority', 'Rainbow DQN'], [50, 800])
    # plot('Reward_DQN + priority memory', 'DQN + priority memory', ['episodes', 'reward'], ['Basic DQN'])
