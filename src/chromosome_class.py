from copy import deepcopy
from instance_class import Instance
from solution_class import Solution
import numpy as np
import matplotlib.pyplot as plt
import random as rd

import networkx as nx
from utils.math_utils import dist, dist_point_to_list
from utils.fifo_queue import put_item


class Chromosome(Solution):
    def __init__(self, instance, list_captors):
        """Class for Chromosome object

        Args:
            instance (Instance) : instance we want to optimize
            list_captors (list of tuples): liste des coordonnées des capteurs dans la solution
        """
        super().__init__(list_captors)
        self.instance = instance  # Instance to solve (targets coordinates, k, Rcom, Rcapt, ...)
        self.valid = False
        self.penalization = 0  # when solution is infeasible, we add a penalization
        self.list_captors = list_captors  # first representation of the information on captors : list of coordinates
        self.captors_binary = list()  # second representation : binary list of length len(instance.targets) + 1

        self.fitness_value = 0
        self.nb_captors = None  # always = len(self.list_captors)

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

        # caution : we do not distinguish the origin if it has a captor on it ?
        for captor in self.list_captors:
            plt.scatter(captor[0], captor[1], marker="D", color='orange')

    @staticmethod
    def draw_uncovered_targets(targets):
        for target in targets:
            plt.scatter(target[0], target[1], marker="+", color='red')

    def display(self, instance, uncovered_targets=None, mark_red=None):
        plt.figure(f"Solution of value {len(self.list_captors) + self.penalization}")
        self.draw_main_info(instance)
        if mark_red:
            plt.scatter(mark_red[0], mark_red[1], marker="+", color='red')
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
        connected_components = [[e for e in c] for c in connected_components]
        return connected_components

    def find_not_covered_targets(self, instance):
        """
            Return the list of targets that are not sufficiently covered (captors < k)
        """
        not_covered = []
        self.disk_graph_targets(instance)
        targets_cover = self.disk_graph_capt.in_degree
        for target in targets_cover:
            if target[1] < instance.k:
                not_covered.append(target[0])

        return not_covered

    def check_with_disk_graph(self, instance):
        """ Check if the solution is feasible, using disk graphs
            Warning : this method is quite expensive (time + memory)"""
        not_covered_targets = self.find_not_covered_targets(instance)
        connected_components = self.find_connected_components(instance)
        return len(not_covered_targets) == 0 and len(connected_components) == 1

    def penalize_infeasibility(self):
        self.disk_graph_captors(self.instance)
        connected_components = nx.connected_components(self.disk_graph_com)
        nb_connected_components = len([c for c in connected_components])

        self.disk_graph_targets(self.instance)
        targets_cover = self.disk_graph_capt.in_degree
        nb_uncovered_targets = 0
        for target in targets_cover:
            nb_uncovered_targets += max(self.instance.k - target[1], 0)

        # nb_uncovered_targets = len(self.find_not_covered_targets(self.instance))
        # nb_connected_components = len(self.find_connected_components(self.instance))
        return nb_uncovered_targets + (nb_connected_components - 1) * int(4 / self.instance.Rcom)

    def find_best_candidate_connectivity(self, instance, connected_components):
        """
            Method used for the Reparation Heuristic
        Args:
            instance (Instance)
            connected_components (list of list of tuples) : list of list of captors (represented with their 2D
                coordinates). Each list of captors is a connected component.
        Returns:
            The "best" target where to put a captor, in order to try to make the solution connected (in terms of comm)
        """
        E_com = instance.neighbours_Rcom

        n_components = len(connected_components)
        i_origin = 0

        if n_components > 1:
            for i in range(n_components):
                # If it is not the component containing the origin
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

    def reparation_heuristic(self, instance, verbose=False):
        """
            Implementation of a Reparation Heuristic
        """

        not_covered_targets = self.find_not_covered_targets(instance)
        if verbose:
            print("")
            print(f"The current solution has {len(not_covered_targets)} targets which are not sufficiently covered.")

        # Init with the farthest captor from the origin (for version 1 only)
        # v = dist_point_to_list((0, 0), self.list_captors)[-1][1]

        # While there are targets which are not k-covered
        while len(not_covered_targets) > 0:
            # v = self.find_best_candidate_covering(instance, v)
            v = self.find_best_candidate_covering_v2(not_covered_targets)
            self.list_captors.append(v)
            not_covered_targets = self.find_not_covered_targets(instance)
            if verbose:
                print(f"- {len(not_covered_targets)} targets not sufficiently covered.")
            # self.display(instance, not_covered_targets)

        connected_components = self.find_connected_components(instance)
        if verbose:
            print(f"The current solution has {len(connected_components)} unconnected groups of captors.")

        # If solution is not connected
        # We go through all components (except the one containing (0,0)) and make them connected
        while len(connected_components) > 1:
            v = self.find_best_candidate_connectivity(instance, connected_components)
            self.list_captors.append(v)
            connected_components = self.find_connected_components(instance)
            if verbose:
                print(f"- {len(connected_components)} unconnected groups of captors.")

        # Update captors attributes
        self.nb_captors = len(self.list_captors)
        self.update_captors_binary()

    def update_list_captors(self):
        """When the binary representation of the captors has been updated but not the list one, we update it manually"""
        self.list_captors = list()
        for i in range(len(self.captors_binary) - 1):
            if int(self.captors_binary[i]) == 1:
                self.list_captors.append(self.instance.targets[i])
        if int(self.captors_binary[-1]) == 1:
            self.list_captors.append((0, 0))

    def update_captors_binary(self):
        """When the list representation of the captors has been updated but not the binary one, we update it manually"""
        n = len(self.instance.targets)
        self.captors_binary = [0 for k in range(n + 1)]  # + 1 for the origin
        for i in range(n):
            if self.instance.targets[i] in self.list_captors:
                self.captors_binary[i] = 1
        if (0, 0) in self.list_captors:
            self.captors_binary[-1] = 1

    def tabu_search(self, size=3, max_iter=30):
        """
            Neighborhood with Hamming distance : uv edge if and only if distance(u,v) == 1
        Args:
            size (int) : size of the tabu list at each iteration
            max_iter (int) : maximum number of iterations
        """
        Ecom = self.instance.neighbours_dict(self.instance.Rcom)
        list_tabu = list()
        targets = self.instance.targets + [(0, 0)]

        candidates = deepcopy(targets)  # points candidates for the modification (0 --> 1 or 1 --> 0)
        i = 0
        # store the best current solution and its associated value + penalization
        best_solution = [deepcopy(self.captors_binary), len(self.list_captors), self.penalization]

        while len(candidates) > 0 and i < max_iter:
            # Chose the transformation to apply

            # M1 : random
            if i == 0:
                random_index = rd.randint(0, len(candidates) - 1)
                modif_target = candidates[random_index]
                # modif_index = targets.index(modif_target)

                neighbours = Ecom[modif_target]
                candidates = deepcopy(neighbours)

            # M2 : best neighbour : (nb_captors, target_modified, penalization)
            best_neighbour_solution = [len(self.instance.targets) + 2, None, 0]
            for target in candidates:
                current_solution = deepcopy(self)
                modif_index = targets.index(target)
                current_solution.captors_binary[modif_index] = 1 - current_solution.captors_binary[modif_index]
                current_solution.update_list_captors()

                value = current_solution.value()
                is_valid = current_solution.is_valid(self.instance)
                pen = 0
                if not is_valid:
                    pen = current_solution.penalize_infeasibility()

                if value + pen < best_neighbour_solution[0] + best_neighbour_solution[2]:
                    best_neighbour_solution[0] = value
                    best_neighbour_solution[1] = target
                    best_neighbour_solution[2] = pen

            modif_target = best_neighbour_solution[1]
            modif_index = targets.index(modif_target)
            best_neighbour_value = best_neighbour_solution[0]
            best_neighbour_pen = best_neighbour_solution[2]
            # self.display(self.instance, mark_red=modif_target)
            self.captors_binary[modif_index] = 1 - self.captors_binary[modif_index]
            self.update_list_captors()

            list_tabu = put_item(list_tabu, modif_target, size)
            if best_neighbour_value + best_neighbour_pen < best_solution[1] + best_solution[2]:
                best_solution[0] = deepcopy(self.captors_binary)
                best_solution[1] = best_neighbour_value
                best_solution[2] = best_neighbour_pen

            neighbours = Ecom[modif_target]
            candidates = deepcopy(neighbours)
            for tabu_target in list_tabu:
                if tabu_target in candidates:
                    candidates.remove(tabu_target)

            i += 1

        return best_solution


class TrivialSolutionRandomized(Chromosome):
    """ Heuristic to generate a Randomized Trivial Solution.
        Used to get an initial population for the evolutionary algorithm"""
    def __init__(self, instance):
        super().__init__(instance, list_captors=None)
        # We consider the trivial solution : all targets get a captor
        self.list_captors = deepcopy(instance.targets)
        np.random.shuffle(self.list_captors)  # shuffle the list of captors
        n = len(instance.targets) // 3
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

        self.update_captors_binary()
