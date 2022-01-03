# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os


def get_png_path(png_name):
    folder_path = os.getcwd().replace('\\', '/')
    png_path = os.path.join(folder_path,'img',png_name)
    return png_path

def plot(list,title,png_name):
    s = len(list)
    x = [i + 1 for i in range(s)]
    y = list

    plt.figure(1)
    # plt.axis([1, s, 0, 1])
    plt.title(title)
    plt.plot(x, y)

    png_path = get_png_path(png_name)
    plt.savefig(png_path)
    plt.show()

def plot_win_rate(list_win_rate):
    return plot(list_win_rate, 'Win rate', 'winrate.png')

def plot_reward(list_reward):
    return plot(list_reward, 'Reward', 'reward.png')


if __name__ == '__main__':

    list_win_rate = [i*0.001/5 for i in range(500)]
    print(list_win_rate)
    plot_win_rate(list_win_rate)
    print('------------------------')

    list_reward = [i*0.001/5 for i in range(500)]
    print(list_reward)
    plot_reward(list_reward)




