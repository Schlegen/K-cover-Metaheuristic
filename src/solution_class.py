from copy import deepcopy
from instance_class import Instance
from utils.errors import Error, InputError
import math as mh
import numpy as np
import matplotlib.pyplot as plt
import random as rd

import networkx as nx
from utils.math_utils import dist, dist_point_to_list


class Solution:

    def __init__(self, list_captors):
        """constructeur de la classe Solution

        Args:
            list_captors (list of tuples): liste des coordonnées des capteurs dans la solution
        """
        self.valid = False
        self.list_captors = list_captors

    def is_valid(self, instance):
        # TODO : l'améliorer pour qu'elle renvoie False très vite selon certains critères (nombre mini de capteurs par
        # ex...) Et dans tous les cas lui faire renvoyer False avant la fin de la boucle quand il en est capable
        # Par ex: au lieu de faire boucler sur tout puis compter, boucler uniquement sur les noeuds qui nous intéressent
        # et retourner faux dès que y en a 1 qui est pas couvert

        """verifie que la solution respecte bien les contraintes de l'instance

        Args:
            instance (Instance): instance à laquelle correspond la solution
        """
        void_grid = deepcopy(instance.grid)

        Rmax = int(max(instance.Rcapt, instance.Rcom))

        # on dispose les capteurs dans la grille :
        for captor in self.list_captors:
            i, j = captor[0], captor[1]
            if i < 0 or j < 0 or j > instance.n or j > instance.m or void_grid[i, j] == 0:
                raise InputError(f"Solution invalide : On ne peut pas placer le capteur ({i},{j})")
            else:
                void_grid[captor[0], captor[1]] = 3

        captors_to_treat = []  # liste des sommets qui communiquent avec le puits
        n_covered_points = 0  # nombre de points couverts par les capteurs

        # initialisation
        for i in range(0, int(instance.Rcom)+1):
            for j in range(0, int(instance.Rcom)+1):
                if i < instance.n and j < instance.m and \
                        mh.sqrt(i ** 2 + j ** 2) <= instance.Rcom and void_grid[i, j] == 3:
                    captors_to_treat.append([i, j])
                    n_covered_points += 1
                    void_grid[i, j] = 0

        # Pour chaque capteur de la liste

        while len(captors_to_treat) > 0:
            captor = captors_to_treat.pop()
            for i in range(-Rmax, Rmax + 1):
                for j in range(-Rmax, Rmax + 1):
                    dist_to_captor = mh.sqrt(i ** 2 + j ** 2)
                    if instance.n > captor[0] + i >= 0 and instance.m > captor[1] + j >= 0:
                        if dist_to_captor <= instance.Rcapt and (void_grid[captor[0] + i, captor[1] + j] == 1 or void_grid[captor[0] + i, captor[1] + j] == 2):
                            n_covered_points += 1
                            void_grid[captor[0] + i, captor[1] + j] = 0
                        elif dist_to_captor <= instance.Rcom and void_grid[captor[0] + i, captor[1] + j] == 3:
                            captors_to_treat.append((captor[0] + i, captor[1] + j))
                            n_covered_points += 1
                            void_grid[captor[0] + i, captor[1] + j] = 0
        self.valid = n_covered_points == instance.n * instance.m - instance.n_deleted_points
        return self.valid

    def value(self):
        return len(self.list_captors)

    def draw_main_info(self, instance):
        # TODO : A enrichir pour afficher les liens de communication et de captation
        for i in range(instance.n):
            for j in range(instance.m):
                if instance.grid[i, j] == 1:
                    plt.scatter(i, j, marker="+", color='blue')
                elif instance.grid[i, j] == 2:
                    plt.scatter(i, j, marker="o", color='black')

        for captor in self.list_captors:
            plt.scatter(captor[0], captor[1], marker="D", color='orange')

    @staticmethod
    def draw_uncovered_targets(targets):
        for target in targets:
            plt.scatter(target[0], target[1], marker="+", color='red')

    def display(self, instance, uncovered_targets=None):
        plt.figure("Solution")
        self.draw_main_info(instance)
        if not self.valid and uncovered_targets is not None:
            self.draw_uncovered_targets(uncovered_targets)
        plt.show()


class TrivialSolution(Solution):

    def __init__(self, instance):
        # We consider the trivial solution : all targets get a captor
        self.list_captors = deepcopy(instance.targets)
        n = len(instance.targets)
        n_captors_deleted = 0
        for i in range(n):
            last_captors_valid = deepcopy(self.list_captors)
            k = i - n_captors_deleted

            # We delete the i_th element of the original list
            self.list_captors.pop(k)
            # We check if it generates a valid solution
            solution_is_valid = self.is_valid(instance)
            if solution_is_valid:
                n_captors_deleted += 1
            else:
                # If it is not, we cancel the deletion and continue
                self.list_captors = deepcopy(last_captors_valid)


class TrivialSolutionRandomized0(Solution):

    def __init__(self, instance):
        # We consider the trivial solution : all targets get a captor
        self.list_captors = deepcopy(instance.targets)
        np.random.shuffle(self.list_captors)  # shuffle the list of captors
        n = len(instance.targets)
        n_captors_deleted = 0
        for i in range(n):
            last_captors_valid = deepcopy(self.list_captors)
            k = i - n_captors_deleted

            # We delete the i_th element of the original list
            self.list_captors.pop(k)
            # We check if it generates a valid solution
            solution_is_valid = self.is_valid(instance)
            if solution_is_valid:
                n_captors_deleted += 1
            else:
                # If it is not, we cancel the deletion and continue
                self.list_captors = deepcopy(last_captors_valid)


class MinCostFlowMethod(Solution):
    def __init__(self, instance):

        flow_value = instance.k * instance.n_points_to_cover
        #print("flow value", flow_value)
        G = nx.DiGraph()

        G.add_node((0, 0, 0), demand=-flow_value)
        G.add_nodes_from([(1, e[0], e[1]) for e in instance.points_to_cover])
        G.add_node((1, 0, 0))  # on peut poser un capteur en 0
        G.add_nodes_from([(2, e[0], e[1]) for e in instance.points_to_cover])
        G.add_node((3, 0, 0), demand=flow_value)

        G.add_edges_from([((0, 0, 0), (1, v[0], v[1])) for v in instance.points_to_cover if dist((0,0), v) <= instance.Rcom])
        G.add_edge((0, 0, 0), (1, 0, 0))

        E_com = instance.neighbours(instance.Rcom, take_origin=True)
        G.add_edges_from([((1, u[0], u[1]), (1, v[0], v[1])) for u, v in E_com], weight=1)

        E_capt = instance.neighbours(instance.Rcapt, take_origin=False)
        G.add_edges_from([((1, u[0], u[1]), (2, v[0], v[1])) for u, v in E_capt], capacity=1)
        G.add_edges_from([((1, u[0], u[1]), (2, u[0], u[1])) for u in instance.points_to_cover], capacity=1)


        G.add_edges_from([((2, v[0], v[1]), (3, 0, 0)) for v in instance.points_to_cover], capacity=instance.k)

        self.graph = G
        self.list_captors = []

    def compute_flow(self):
        self.flow = nx.algorithms.flow.min_cost_flow(self.graph)

    def build_solution(self):
        self.list_captors = []
        n_captors = 0
        for key, value in self.flow[(0, 0, 0)].items():
            if value >= 1:
                self.list_captors.append((key[1], key[2]))
                n_captors += 1
    
        n_treated_captors = 0
        while n_treated_captors < n_captors:
            #time.sleep(0.1)
            captor = self.list_captors[n_treated_captors]
            n_treated_captors += 1
            
            for key, value in self.flow[(1, captor[0], captor[1])].items():
                #print(captor, "-->", key, value)
                if key[0] == 1 and value >= 1:
                    self.list_captors.append((key[1], key[2]))
                    n_captors += 1
            
        return 0