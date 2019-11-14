import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


def plot_result(scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    pd.DataFrame(scores).mean(axis=1).rolling(100, min_periods=1).mean().plot(ax=ax1, color='b')
    ax1.plot(np.arange(len(scores)), np.repeat(30, len(scores)), color='y')
    ax1.legend(['Moving Avg (100 episodes)', 'Target'])
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Episode #')
    ax1.set_title('Moving Average (100 episodes window) ')

    ax2.plot(np.arange(len(scores)), np.mean(scores, axis=1), color='b')
    ax2.plot(np.arange(len(scores)), np.max(scores, axis=1), color='g')
    ax2.plot(np.arange(len(scores)), np.min(scores, axis=1), color='r')
    ax2.plot(np.arange(len(scores)), np.repeat(30, len(scores)), color='y')
    ax2.fill_between(np.arange(len(scores)), np.max(scores, axis=1), np.min(scores, axis=1), alpha=0.5)
    ax2.legend(['Mean', 'Max', 'Min', 'Target'])
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Episode #')
    ax2.set_title('Episodes Statistics')


def path_result_folder(file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results',  file_name)