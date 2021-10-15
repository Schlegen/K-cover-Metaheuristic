import matplotlib.pyplot as plt
import numpy as np


def display(grid):
    # fig = plt.figure()
    x, y = grid.shape
    for i in range(x):
        for j in range(y):
            if grid[i, j] == 1:
                print((i, j))
                plt.scatter(i, j, marker="+", color='blue')
            elif grid[i, j] == 2:
                plt.scatter(i, j, marker="o", color='red')
    plt.show()

