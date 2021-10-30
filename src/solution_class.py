from copy import deepcopy
from operator import ne
from numpy.lib.function_base import insert, select
from instance_class import Instance
from utils.errors import Error, InputError
from utils.math_utils import subgraph_is_connex, n_connex_components
import math as mh
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random as rd
import time
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
        """verifie que la solution respecte bien les contraintes de l'instance

        Args:
            instance (Instance): instance à laquelle correspond la solution
        """

        self.solution = np.zeros(instance.n_targets+1, dtype=np.int64)
        self.solution[0] = 1
        for captor in self.list_captors:
            self.solution[instance.indexes[captor]] = 1

        self.coverage_vect = instance.k * np.ones(instance.n_targets+1, dtype=np.int64)
        self.coverage_vect[0] = 0

        for v in np.argwhere(self.solution.flatten()).flatten():
            self.coverage_vect -= instance.E_capt[v].flatten()

        res = np.all(self.coverage_vect <= 0)

        if not res:
            self.valid = False
        
        else:
            self.valid = subgraph_is_connex(instance.E_com, np.argwhere(self.solution).flatten())

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

class LocalSearch(Solution):

    def __init__(self, instance):
        self.instance = instance
        #construction de la matrice d'adjacence de 
        self.list_captors = []

    def GenerateInitialSolution(self):
        self.list_captors = []
        
        self.solution = np.zeros(self.instance.n_targets+1, dtype=np.int64)
        self.solution[0] = 1

        #tableau notant le nombre de capteurs manquant à chaque cible (peut être négatif en cas d'exces)
        self.coverage_vect = self.instance.k * np.ones(self.instance.n_targets+1, dtype=np.int64)
        self.coverage_vect[0] = 0

        while np.any(self.coverage_vect > 0):

            remaining_degrees_com = ((1 - self.solution.reshape(1, self.instance.n_targets+1)) @ self.instance.E_com).flatten()
            remaining_degrees_capt = (np.minimum(1, np.maximum(0, self.coverage_vect.reshape(1, self.instance.n_targets+1))) @ self.instance.E_capt).flatten()
            remaining_degrees = remaining_degrees_com + remaining_degrees_capt

            connex_neighbours = np.minimum(1, (self.solution.reshape(1, self.instance.n_targets+1) @ self.instance.E_com).flatten())
            selected_captor = np.argmax(connex_neighbours * remaining_degrees * (1 - self.solution))
            self.solution[selected_captor] = 1

            # mise à jour des sommets captés
            self.coverage_vect -= self.instance.E_capt[selected_captor].flatten()

        for u in np.argwhere(self.solution).flatten():
            self.list_captors.append(self.instance.reversed_indexes[u])

    def improve_solution(self):
        self.list_captors = []
        # suppression des capteurs que l'on peut enlever (algo glouton)
        for u in np.argwhere(self.solution).flatten():
            if u > 0:
                can_remove = True
    
                # on regarde si les cibles captées seraient suffisemment captées sans le capteur
                list_capt = np.argwhere(self.instance.E_capt[u].flatten()).flatten()
                n_capted = len(list_capt)
                i = 0
                while (i < n_capted) and can_remove:
                    v = list_capt[i]
                    can_remove = self.coverage_vect[v] < 0
                    i += 1

                # ensuite on regarde si la connexité est conservée
                solution_without_u = deepcopy(self.solution)
                solution_without_u[u] = 0
                can_remove = can_remove and subgraph_is_connex(self.instance.E_com, np.argwhere(solution_without_u).flatten())

                # on enleve les sommets et on met à jour les sommets captés
                if can_remove:
                    self.solution[u] = 0
                    for target in list_capt:
                        self.coverage_vect[target] += 1

                else:
                    self.list_captors.append(self.instance.reversed_indexes[u])

    def delete_capt(self, solution, coverage_vect, v):
        """ supprime un capteur et met à jour la solution associée
        s'assurer qu'un capteur est bien placé à l'indice correspondant

        Args:
            solution (np.array): vecteur de taille n_targets+1 avec des 1 aux emplacements des capteurs
            coverage_vect (np.array): vecteur de taille n_targets+1 stockant le nombre de capteurs captant chaque sommet
            v (int): indice de l'emplacement du capteur sortant

        modifie solution et coverage_vect
        """

        solution[v] = 0
        coverage_vect -= self.instance.E_capt[v].flatten()

    def add_capt(self, solution, coverage_vect, v):
        """ ajoute un capteur et met à jour la solution associée
        s'assurer qu'aucun capteur n'est placé à l'indice correspondant

        Args:
            solution (np.array): vecteur de taille n_targets+1 avec des 1 aux emplacements des capteurs
            coverage_vect (np.array): vecteur de taille n_targets+1 stockant le nombre de capteurs captant chaque sommet
            v (int): indice de l'emplacement du capteur entrant

        modifie solution et coverage_vect
        """

        solution[v] = 1
        coverage_vect += self.instance.E_capt[v].flatten()

    def exchange11(self, solution, coverage_vect, v_out, v_in):
        """ transformation deplace un capteur en v_in vers v_out
            il faut s'assurer en amont que la solution possede un capteur en v_out et pas en v_in

        Args:
            solution (np.array): vecteur de taille n_targets+1 avec des 1 aux emplacements des capteurs
            coverage_vect (np.array): vecteur de taille n_targets+1 stockant le nombre de capteurs captant chaque sommet
            v_out (int): indice de l'emplacement du capteur sortant
            v_in (int): indice de l'emplacement du capteur entrant

        modifie solution et coverage_vect
        """
        self.delete_capt(solution, coverage_vect, v_out)
        self.add_capt(solution, coverage_vect, v_in)

    def exchange22(self, solution, coverage_vect, v_out1, v_out2, v_in1, v_in2):
        """ transformation deplace deux capteurs de v_out1 et v_out2 vers v_in1, v_in2
            il faut s'assurer en amont que la solution possede des capteur en v_out1 et v_out2 et pas en v_in1 ni v_in2

        Args:
            solution (np.array): vecteur de taille n_targets+1 avec des 1 aux emplacements des capteurs
            coverage_vect (np.array): vecteur de taille n_targets+1 stockant le nombre de capteurs captant chaque sommet
            v_out (int): indice de l'emplacement du capteur sortant
            v_in (int): indice de l'emplacement du capteur entrant

        modifie solution et coverage_vect
        """

        self.delete_capt(solution, coverage_vect, v_out1)
        self.delete_capt(solution, coverage_vect, v_out2)
        self.add_capt(solution, coverage_vect, v_in1)
        self.add_capt(solution, coverage_vect, v_in2)

    def exchange21(self, solution, coverage_vect, v_out1, v_out2, v_in1):
        self.delete_capt(solution, coverage_vect, v_out1)
        self.delete_capt(solution, coverage_vect, v_out2)
        self.add_capt(solution, coverage_vect, v_in1)

    def exchange12(self, solution, coverage_vect, v_out1, v_in1, v_in2):
        self.delete_capt(solution, coverage_vect, v_out1)
        self.add_capt(solution, coverage_vect, v_in1)
        self.add_capt(solution, coverage_vect, v_in2)

    def fitness_solution(self, solution, coverage_vect, pen_capt, pen_connexity):
        return np.sum(solution) + pen_capt * np.linalg.norm(np.maximum(0, np.coverage_matrix), ord=1) + pen_connexity * n_connex_components(self.instance.E_com, solution)

    def best_in_neighbourhood(self, n_neighbours, p11, p22, p21):
        # p12 = 1 - p11 - p21 - p12 
        # peut bugguer si on ne peut pas choisir les sommets hors et dans l'arbre
        best_fitness = np.inf
        for i in range(n_neighbours):
            new_solution = deepcopy(self.solution)
            new_coverage = deepcopy(self.coverage_matrix)

            choice = np.random.choice(np.arrange(1, 5), [p11, p22, p21, 1 - p11 - p22 - p21])

            if choice == 1:
                v_out = np.random.choice(np.argwhere(self.solution[1:])) + 1
                v_in = np.random.choice(1 - np.argwhere(self.solution[1:])) + 1
                self.exchange11(new_solution, new_coverage, v_out, v_in)

            elif choice == 2:
                v_out = np.random.choice(np.argwhere(self.solution[1:]), 2, replace=False) + 1
                v_in = np.random.choice(1 - np.argwhere(self.solution[1:]), 2, replace=False) + 1
                self.exchange22(new_solution, new_coverage, v_out[0], v_out[1], v_in[0], v_in[1])

            elif choice == 3:
                v_out = np.random.choice(np.argwhere(self.solution[1:]), 2, replace=False) + 1
                v_in = np.random.choice(1 - np.argwhere(self.solution[1:])) + 1
                self.exchange21(new_solution, new_coverage, v_out[0], v_out[1], v_in)

            else:
                v_out = np.random.choice(np.argwhere(self.solution[1:])) + 1
                v_in = np.random.choice(1 - np.argwhere(self.solution[1:]), 2, replace=False) + 1
                self.exchange12(new_solution, new_coverage, v_out, v_in[0], v_in[1])

            new_fitness = self.fitness_solution(new_solution, new_coverage)

            if new_fitness < best_fitness:
                best_solution = deepcopy(new_solution)
                best_coverage = deepcopy(new_coverage)
                best_fitness = new_fitness
            
        return best_solution, best_coverage, best_fitness

class TabuSearch(LocalSearch):
    def __init__(self,instance):
        super().__init__(instance)

    def search(self, n_max_without_improvement, time_limit):
        self.GenerateInitialSolution()
        self.improve_solution()
        begin = time.time()

        while time.time() - begin < time_limit:
            ()


