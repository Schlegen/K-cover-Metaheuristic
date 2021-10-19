import numpy as np


def extract_points(data_file):
    content = [i.strip().split() for i in open(data_file).readlines()]
    points = [(int(x[2][1:-1]), int(x[3][:-1])) for x in content[2:]]
    return points


def generate_grid(size):
    grid = np.ones((size[0], size[1]))
    grid[0, 0] = 2
    return grid


def delete_captors(grid, coordinates):
    for point in coordinates:
        grid[point] = 0
    return grid


def init_grid(points_to_delete, size):
    grid = generate_grid(size)
    grid = delete_captors(grid, points_to_delete)
    return grid

def coordinates_to_cover(grid):
    list_coordinates = []
    n, m = grid.shape
    for i in range(n):
        for j in range(m):
            if grid[i,j] == 1:
                list_coordinates.append((i,j))
    return list_coordinates
