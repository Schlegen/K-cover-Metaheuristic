from utils.build_grid import extract_points, init_grid, coordinates_to_cover, extract_points_random
from utils.math_utils import dist
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class Instance:
    def __init__(self, particular_points, size, Rcapt=1, Rcom=1, k=1, with_float=True):
        """constructeur de la classe Instance

        Args:
            Rcapt (float, optionnal): rayon de captage. Defaults to 1.
            Rcom (float, optionnal): rayon de communication. Defaults to 1.
            particular_points (list of tuples) : may be the points to delete from a grid (with_float=False) or
                the coordinates of the targets (with_float=False)
            with_float (boolean) : tells whether we are working with randomly generated points or with discrete grid
        """

        self.source = (0, 0)
        self.k = k

        if with_float:  # if the instance is cloud of randomly generated points
            self.targets = particular_points
        else:  # else, the instance is a squared grid
            self.n = size[0]
            self.m = size[1]

            self.deleted_points = particular_points
            self.n_deleted_points = len(particular_points)

            self.targets = [(i, j) for i in range(self.n) for j in range(self.m) if (i, j) not in particular_points
                            and (i, j) != self.source]
            self.grid = init_grid(particular_points, size)

        self.n_targets = len(self.targets)

        self.Rcapt = Rcapt
        self.Rcom = Rcom

        self.neighbours_Rcapt = self.neighbours_dict(Rcapt)
        self.neighbours_Rcom = self.neighbours_dict(Rcom)

        #utilisés dans la classe tabou
        self.indexes = {e : i+1 for i,e in enumerate(sorted(self.targets))}
        self.indexes[(0, 0)] = 0

        self.reversed_indexes = {i+1 : e for i,e in enumerate(sorted(self.targets))}
        self.reversed_indexes[0] = (0, 0)

        # construction de la matrice d'adjacence de captation
        capt_neighbours = self.neighbours(self.Rcapt, take_origin=False)
        self.E_capt = np.eye(self.n_targets+1, dtype=np.int8)
        for arc in capt_neighbours:
            self.E_capt[self.indexes[arc[0]], self.indexes[arc[1]]] = 1

        # construction de la matrice d'adjacence de communication
        com_neighbours = self.neighbours(self.Rcom, take_origin=True)
        self.E_com = np.zeros((self.n_targets+1, self.n_targets+1), dtype=np.int8)
        for arc in com_neighbours:
            self.E_com[self.indexes[arc[0]], self.indexes[arc[1]]] = 1

    @classmethod
    def from_disk(cls, data_file, Rcapt=1, Rcom=1, k=1, with_float=True):
        if with_float:
            points = extract_points_random(data_file)
            size = len(points)
        else:
            points, size = extract_points(data_file)
        return cls(points, size, Rcapt, Rcom, k)

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
            for i in range(self.n_targets):
                u = self.targets[i]
                if dist(u, v) <= R:
                    list_neighbours.append((u, v))
                    list_neighbours.append((v, u))

        for i in range(self.n_targets):
            u = self.targets[i]
            for j in range(i):
                v = self.targets[j]
                if dist(u, v) <= R:
                    list_neighbours.append((u, v))
                    list_neighbours.append((v, u))
        return list_neighbours

    def neighbours_dict(self, R):
        """Renvoie le dictionnaire des voisins

        Args:
            R (float): rayon
            take_origin (bool): indique si l'on doit prendre le point (0, 0, 0 dans le calcul)

        Returns:
            liste de tuples de tuples: chaque element est un couple de points de la liste
        """
        list_neighbours = dict()

        targets_with_origin = self.targets + [(0, 0)]

        for i in range(self.n_targets + 1):
            u = targets_with_origin[i]
            if u not in list_neighbours:
                list_neighbours[u] = list()
            for j in range(i):
                v = targets_with_origin[j]
                if v not in list_neighbours:
                    list_neighbours[v] = list()
                if dist(u, v) <= R:
                    list_neighbours[u].append(v)
                    list_neighbours[v].append(u)

        return list_neighbours
