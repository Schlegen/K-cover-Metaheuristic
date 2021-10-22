import numpy as np


def extract_points(data_file):
    content = [i.strip() for i in open(data_file).readlines()]
    size = int(np.sqrt(int(content[0].split(":=")[1].split(";")[0])))
    points = [(int(x.split('(')[1].split(",")[0]),
               int(x.split(',')[1].split(")")[0])) for x in content[3:]]
    return points, (size, size)


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
            if grid[i, j] == 1:
                list_coordinates.append((i, j))
    return list_coordinates
