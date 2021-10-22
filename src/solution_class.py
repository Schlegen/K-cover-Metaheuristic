from copy import deepcopy
from instance_class import Instance
from utils.errors import Error, InputError
import math as mh
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from utils.math_utils import dist


class Solution:

    def __init__(self, list_captors):
        """constructeur de la classe Solution

        Args:
            list_captors (list of tuples): liste des coordonnées des capteurs dans la solution
        """
        self.list_captors = list_captors
        self.captors = None

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
        self.captors = np.array(
            [[int((i, j) in self.list_captors) for i in range(instance.n)] for j in range(instance.m)]
        )
        # je pense qu'il faudra tot ou tard travailler avec un tableau binaire comme ca,
        # qui represente la variable x binaire du pb lineaire

        Rmax = int(max(instance.Rcapt, instance.Rcom))

        # on dispose les capteurs dans la grille :
        for captor in self.list_captors:
            i, j = captor[0], captor[1]
            if i < 0 or j < 0 or j > instance.n or j > instance.m or void_grid[i, j] == 0:
                raise InputError(f"Solution invalide : On ne peut pas placer le capteur ({i},{j})")
            else:
                void_grid[captor[0], captor[1]] = 3

                self.captors[i] = 1

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
        return n_covered_points == instance.n * instance.m - instance.n_deleted_points

    def value(self):
        return len(self.list_captors)

    def display(self, instance):
        # TODO : A enrichir pour afficher les liens de communication et de captation

        plt.figure("Solution")
        for i in range(instance.n):
            for j in range(instance.m):
                if instance.grid[i, j] == 1:
                    plt.scatter(i, j, marker="+", color='blue')
                elif instance.grid[i, j] == 2:
                    plt.scatter(i, j, marker="o", color='red')

        for captor in self.list_captors:
            plt.scatter(captor[0], captor[1], marker="D", color='orange')
        plt.show()

    def disk_graph_captors(self, instance):
        """
            Generates the disk graph of captors, with radius = Rcom
        """
        G = nx.Graph()
        points_to_communicate_with = self.list_captors + [(0, 0)]
        G.add_nodes_from([(e[0], e[1]) for e in points_to_communicate_with])

        E_com = instance.neighbours_dict(instance.Rcom)
        for u in points_to_communicate_with:
            for v in E_com[u]:
                if v in points_to_communicate_with:
                    G.add_edge((u[0], u[1]), (v[0], v[1]))
            # G.add_edges_from([((u[0], u[1]), (v[0], v[1])) for v in E_com[u]])
        self.disk_graph_com = G

    def disk_graph_targets(self, instance):
        """
            Generates the disk graph of targets, with radius = Rcapt
        """
        G = nx.Graph()
        points_to_capt = self.list_captors
        G.add_nodes_from([(e[0], e[1]) for e in points_to_capt])

        E_capt = instance.neighbours_dict(instance.Rcapt)
        for u in points_to_capt:
            G.add_edges_from([((u[0], u[1]), (v[0], v[1])) for v in E_capt[u]])
        self.disk_graph_capt = G

    def find_connected_components(self, instance):
        self.disk_graph_captors(instance)
        connected_components = nx.connected_components(self.disk_graph_com)
        print("Captors")
        print(self.list_captors)
        print("Connected components")
        for c in connected_components:
            print([e for e in c])
        print("---")
        return connected_components

    def reparation_heuristic(self, instance):
        """
            Implementation d'une heuristique de reparation
        :param instance:
        :return:
        """
        # Si des capteurs sont pas captés
        # TODO

        # Si la solution non connexe
        # On parcourt les composantes connexes des capteurs (sauf celle qui contient (0,0) ), et on les rend connexes
        # à celle qui contient (0, 0)
        

        return


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


class TrivialSolutionRandomized(Solution):

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

        G.add_node((0,0,0), demand=-flow_value)
        G.add_nodes_from([(1, e[0], e[1]) for e in instance.points_to_cover])
        G.add_node((1, 0, 0)) # on peut poser un capteur en 0
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