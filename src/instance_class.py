from utils.build_grid import extract_points, init_grid, coordinates_to_cover
from utils.math_utils import dist
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class Instance:
    def __init__(self, deleted_points, size, Rcapt=1, Rcom=1, k=1):
        """constructeur de la classe Instance

        Args:
            Rcapt (float, optionnal): rayon de captage. Defaults to 1.
            Rcom (float, optionnal): rayon de communication. Defaults to 1.
        """

        self.source = (0, 0)
        self.n = size[0]
        self.m = size[1]

        self.k = k

        self.deleted_points = deleted_points
        self.n_deleted_points = len(deleted_points)

        self.targets = [(i, j) for i in range(self.n) for j in range(self.m) if (i, j) not in deleted_points
                        and (i, j) != self.source]
        self.n_targets = len(self.targets)

        self.grid = init_grid(deleted_points, size)

        self.points_to_cover = coordinates_to_cover(self.grid)
        self.n_points_to_cover = len(coordinates_to_cover(self.grid))

        self.Rcapt = Rcapt
        self.Rcom = Rcom

    @classmethod
    def from_disk(cls, data_file, size=(10, 10), Rcapt=1, Rcom=1, k=1):
        captors_to_delete = extract_points(data_file)
        return cls(captors_to_delete, size, Rcapt, Rcom, k)

    def draw_data(self):
        for target in self.targets:
            plt.scatter(target[0], target[1], marker="+", color='blue')
        plt.scatter(self.source[0], self.source[1], marker="o", color='red')

    def display(self):
        plt.figure("Instance")
        self.draw_data()
        plt.show()

    def neighbours(self, R, take_origin):
        """calcule l'ensemble des couples de points à couvrir distants d'au plus R

        Args:
            R (float): rayon
            take_origin (bool): indique si l'on doit prendre le point (0, 0, 0 dans la calcul)

        Returns:
            liste de tuples de tuples: chaque element est un couple de points de la liste 
        """
        list_neighbours = []

        if take_origin:
            v = (0, 0)
            for i in range(self.n_points_to_cover):
                u = self.points_to_cover[i]
                if dist(u, v) <= R:
                    list_neighbours.append((u, v))
                    list_neighbours.append((v, u))

        for i in range(self.n_points_to_cover):
            u = self.points_to_cover[i]
            for j in range(i):
                v = self.points_to_cover[j]
                if dist(u, v) <= R:
                    list_neighbours.append((u, v))
                    list_neighbours.append((v, u))

        return list_neighbours
