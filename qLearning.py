import numpy as np
import random
import sys

sys.setrecursionlimit(15000)

reward_table = [[1,  1,  2,  5,  11, 17, 11,  21,  18,  21,  19,  20,  10,  7,   5,  11, 4,  3,  3,  1],
                [9,  8,  11, 17, 27, 24, 39,  43,  49,  55,  58,  37,  41,  25,  25, 23, 13, 6,  1,  4],
                [10, 16, 24, 38, 35, 59, 77,  85,  109, 97,  82,  81,  63,  58,  50, 33, 14, 13, 10, 9],
                [18, 11, 36, 44, 69, 78, 111, 112, 140, 144, 129, 131, 96,  74,  69, 48, 36, 19, 11, 5],
                [11, 17, 31, 54, 77, 78, 110, 130, 144, 161, 136, 140, 104, 102, 73, 47, 28, 22, 8,  6],
                [10, 19, 25, 40, 67, 79, 90,  140, 125, 129, 154, 135, 106, 89,  66, 49, 22, 20, 9,  7],
                [9,  12, 22, 21, 48, 56, 71,  77,  86,  102, 96,  90,  74,  48,  39, 25, 12, 16, 6,  2],
                [3,  7,  4,  16, 16, 36, 34,  37,  57,  51,  46,  46,  38,  31,  31, 14, 15, 7,  1,  2],
                [1,  2,  2,  9,  7,  9,  15,  15,  14,  11,  20,  18,  17,  12,  13, 7,  4,  3,  2,  0],
                [0,  2,  2,  3,  0,  7,  3,   4,   9,   3,   5,   4,   3,   4,   1,  1,  1,  0,  1,  0]]

reward_table = 200 - np.array(reward_table)

iteration = 10000
gamma = 0.8
learning_rate = 0.7

q_table = np.zeros((len(reward_table), len(reward_table[0])))


def find(reward_table, iteration, gamma, learning_rate, q_table, menzil=20):
    if menzil == 0:
        #menzil = round(random.uniform(1, len(reward_table[0])))
        menzil=20
    next_irtifa = round(random.uniform(0, len(reward_table)-1))
    if menzil == 1:
        q_table[next_irtifa, menzil-1] = q_table[next_irtifa, menzil-1] + learning_rate * (reward_table[next_irtifa][menzil-1] - q_table[next_irtifa, menzil-1])
    else:
        q_table[next_irtifa, menzil-1] = q_table[next_irtifa, menzil-1] + learning_rate * (reward_table[next_irtifa][menzil-1] + gamma * q_table.max(axis=0)[menzil-2] - q_table[next_irtifa, menzil-1])
    if iteration > 1:
        iteration -= 1
        find(reward_table, iteration, gamma, learning_rate, q_table, menzil-1)
    return q_table


print(find(reward_table, iteration, gamma, learning_rate, q_table))