from copy import deepcopy
from instance_class import Instance
from solution_class import Solution
from utils.errors import Error, InputError
import math as mh
import numpy as np
import matplotlib.pyplot as plt
import random as rd

import networkx as nx
from utils.math_utils import dist, dist_point_to_list


class Chromosome(Solution):
    def __init__(self, instance, list_captors):
        """constructeur de la classe Chromosome

        Args:
            list_captors (list of tuples): liste des coordonnées des capteurs dans la solution
        """
        super().__init__(list_captors)
        self.instance = instance
        self.valid = False
        self.list_captors = list_captors
        self.captors_binary = [0 for i in range(len(instance.targets) + 1)]  # + 1 for the origin
        self.fitness_value = 0
        self.nb_captors = None

    def value(self):
        return len(self.list_captors)

    def draw_main_info(self, instance):
        # TODO : A enrichir pour afficher les liens de communication et de captation

        plt.figure("Solution")
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
        G = nx.DiGraph()
        G.add_nodes_from([(e[0], e[1]) for e in instance.targets + [(0, 0)]])

        E_capt = instance.neighbours_Rcapt
        for u in self.list_captors:
            G.add_edge((u[0], u[1]), (u[0], u[1]))
            G.add_edges_from([((u[0], u[1]), (v[0], v[1])) for v in E_capt[u]])
        self.disk_graph_capt = G

    def find_connected_components(self, instance):
        """
        Return the list of connected compononents.
        A connected component is a list of captors that all have a captor at a distance < R_com within the component
        """
        self.disk_graph_captors(instance)
        connected_components = nx.connected_components(self.disk_graph_com)
        # print("Captors")
        # print(self.list_captors)
        # print("Connected components")
        connected_components = [[e for e in c] for c in connected_components]
        # print(connected_components)
        return connected_components

    def find_not_covered_targets(self, instance):
        not_covered = []
        self.disk_graph_targets(instance)
        targets_cover = self.disk_graph_capt.in_degree
        for target in targets_cover:
            if target[1] < instance.k:
                not_covered.append(target[0])

        return not_covered

    def check_with_disk_graph(self, instance):
        not_covered_targets = self.find_not_covered_targets(instance)
        connected_components = self.find_connected_components(instance)
        self.valid = len(not_covered_targets) == 0 and len(connected_components) == 1
        return self.valid

    def find_best_candidate_connectivity(self, instance, connected_components):
        E_com = instance.neighbours_Rcom

        n_components = len(connected_components)
        i_origin = 0

        if n_components > 1:
            for i in range(n_components):
                # S'il ne s'agit pas de la composante contenant l'origine
                if (0, 0) in connected_components[i]:
                    i_origin = i
            for i in range(n_components):
                if i != i_origin:
                    component = connected_components[i]
                    component_sorted = dist_point_to_list((0, 0), component)
                    for u in component_sorted:
                        u_coordinates = u[1]
                        dist_component_origin_to_u = dist_point_to_list(u_coordinates,
                                                                        connected_components[i_origin])
                        for j in range(len(dist_component_origin_to_u)):
                            nearest_point = dist_component_origin_to_u[j][1]

                            nearest_point_neighbours = dist_point_to_list(u_coordinates, E_com[nearest_point])
                            for k in range(len(nearest_point_neighbours)):
                                v_candidate = nearest_point_neighbours[k][1]
                                if v_candidate not in self.list_captors:
                                    return v_candidate

    def find_best_candidate_covering(self, instance, u):
        E_com = instance.neighbours_Rcom
        farthest_neighbours = dist_point_to_list(u, E_com[u])
        farthest_neighbours.reverse()
        for k in range(len(farthest_neighbours)):
            v_candidate = farthest_neighbours[k][1]
            if v_candidate not in self.list_captors:
                return v_candidate

        # if we do not find any relevant candidate, we take one randomly
        while True:
            i_captor = rd.randint(0, len(self.list_captors) - 1)
            captor = self.list_captors[i_captor]
            i_neighbour = rd.randint(0, len(E_com[captor]) - 1)
            v_candidate = E_com[captor][i_neighbour]
            if v_candidate not in self.list_captors:
                return v_candidate

    @staticmethod
    def find_best_candidate_covering_v2(not_covered_targets):
        i = rd.randint(0, len(not_covered_targets) - 1)
        return not_covered_targets[i]

    def reparation_heuristic(self, instance):
        """
            Implementation d'une heuristique de reparation
        """

        not_covered_targets = self.find_not_covered_targets(instance)
        print("")
        print(f"The current solution has {len(not_covered_targets)} targets which are not sufficiently covered.")
        self.display(instance, not_covered_targets)

        # init with the farthest captor from the origin (for version 1 only)
        # v = dist_point_to_list((0, 0), self.list_captors)[-1][1]

        # Tant que des cibles ne sont pas captées
        while len(not_covered_targets) > 0:
            # v = self.find_best_candidate_covering(instance, v)
            v = self.find_best_candidate_covering_v2(not_covered_targets)
            self.list_captors.append(v)
            not_covered_targets = self.find_not_covered_targets(instance)
            print(f"- {len(not_covered_targets)} targets not sufficiently covered.")
            # self.display(instance, not_covered_targets)

        connected_components = self.find_connected_components(instance)
        print(f"The current solution has {len(connected_components)} unconnected groups of captors.")

        # Si la solution non connexe
        # On parcourt les composantes connexes des capteurs (sauf celle qui contient (0,0) ), et on les rend connexes
        # à celle qui contient (0, 0)

        while len(connected_components) > 1:
            v = self.find_best_candidate_connectivity(instance, connected_components)
            self.list_captors.append(v)
            connected_components = self.find_connected_components(instance)
            print(f"- {len(connected_components)} unconnected groups of captors.")

        return 0


class TrivialSolutionRandomized(Chromosome):
    def __init__(self, instance):
        super().__init__(instance, list_captors=None)
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
            solution_is_valid = self.is_valid(self.instance)
            if solution_is_valid:
                n_captors_deleted += 1
            else:
                # If it is not, we cancel the deletion and continue
                self.list_captors = deepcopy(last_captors_valid)

        for i in range(n):
            if instance.targets[i] in self.list_captors:
                self.captors_binary[i] = 1
        if (0, 0) in self.list_captors:
            self.captors_binary[-1] = 1
