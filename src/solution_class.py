from copy import deepcopy
from operator import ne
from numpy.lib.function_base import insert, select
from instance_class import Instance
from utils.errors import Error, InputError, ConvergenceError
from utils.math_utils import subgraph_is_connex, n_connex_components, spanning_tree_subgraph
import math as mh
import numpy as np
import matplotlib.pyplot as plt
from utils.display_utils import circles
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

        return self.valid

    def value(self):
        return len(self.list_captors)

    def draw_main_info(self, instance, ax):
        ax.scatter(0, 0, marker="o", color='black')
        for target in instance.targets:
            ax.scatter(target[0], target[1], marker="+", color='blue')
        for captor in self.list_captors:
            ax.scatter(captor[0], captor[1], marker="D", color='orange')

    def draw_communication(self, instance, ax):
        solution = np.zeros(instance.n_targets+1, dtype=np.int64)
        solution[0] = 1
        for captor in self.list_captors:
            solution[instance.indexes[captor]] = 1

        com_edges = spanning_tree_subgraph(instance.E_com, np.argwhere(solution).flatten())

        for edge in com_edges :
            u, v = instance.reversed_indexes[edge[0]], instance.reversed_indexes[edge[1]]
            ax.plot([u[0], v[0]], [u[1], v[1]], c="grey")

    def display(self, instance):
        fig = plt.figure(f"Solution")
        ax = fig.add_subplot(111)
        ax.set_title(f"value : {self.value()} (Rcapt={instance.Rcapt} Rcom={instance.Rcom} k={instance.k})")
        self.draw_communication(instance, ax)
        out = circles([t[0] for t in self.list_captors], [t[1] for t in self.list_captors], [instance.Rcapt for t in self.list_captors], ax, c="green", alpha=0.1, edgecolor='none')
        self.draw_main_info(instance, ax)
        plt.gca().set_aspect('equal', adjustable='box')
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
        # construction de la matrice d'adjacence de
        self.list_captors = []

    def GenerateInitialSolution(self):
        """heuristique gloutonne qui construit une solution

        Raises:
            ConvergenceError: erreur quand on ne peut pas trouver de solution (instance infaisable)

        Returns:
            solution, coverage: solution stoque les solution avec un vecteur de 0 ou 1, 
                            coverage est un vecteur stockant le nombre de fois où chaque
                            sommet est couvert
        """        
        solution = np.zeros(self.instance.n_targets+1, dtype=np.int64)
        solution[0] = 1

        # tableau notant le nombre de capteurs manquant à chaque cible (peut être négatif en cas d'exces)
        coverage_vect = self.instance.k * np.ones(self.instance.n_targets+1, dtype=np.int64)
        coverage_vect[0] = 0
        i = 0
        while np.any(coverage_vect > 0):
            remaining_degrees_com = ((1 - solution.reshape(1, self.instance.n_targets+1)) @ self.instance.E_com).flatten()
            remaining_degrees_capt = (np.minimum(1, np.maximum(0, coverage_vect.reshape(1, self.instance.n_targets+1))) @ self.instance.E_capt).flatten()
            
            remaining_degrees = remaining_degrees_com + remaining_degrees_capt

            connex_neighbours = np.minimum(1, (solution.reshape(1, self.instance.n_targets+1) @ self.instance.E_com).flatten())
            selected_captor = np.argmax(connex_neighbours * remaining_degrees * (1 - solution))
            if selected_captor == 0:
                print(solution)
                raise ConvergenceError("Unable to generate a feasible initial solution")

            solution[selected_captor] = 1


            # mise à jour des sommets captés
            coverage_vect -= self.instance.E_capt[selected_captor].flatten()

            i += 1
        
        return solution, coverage_vect

    def coverage_as_matrix(self, coverage):
        matrix = np.zeros((self.instance.n, self.instance.m), dtype=int)
        for u, val in enumerate(coverage):
            coords = self.instance.reversed_indexes[u]
            matrix[coords[0], coords[1]] = val
        
    def set_solution(self, solution, coverage_vect):
        """enregistre et stocke une solution

        Args:
            solution (np.array 1D): vecteur stockant la solution avec des 1 aux emplacements occupés
            coverage_vect (np.array 1D): vecteur stockant le nombre de fois où chaque sommet est couvert
        """        
        self.solution = deepcopy(solution)
        self.coverage_vect = deepcopy(coverage_vect)
        self.list_captors = []
        for u in np.argwhere(self.solution[1:]).flatten():
            self.list_captors.append(self.instance.reversed_indexes[u + 1])

    def improve_solution(self, solution, coverage_vect):
        """améliore une solution en enlevant les capteurs que l'on peut enlever

        Args:
            solution (np.array 1D): vecteur stockant la solution avec des 1 aux emplacements occupés
            coverage_vect (np.array 1D): vecteur stockant le nombre de fois où chaque sommet est couvert
        """        
        # suppression des capteurs que l'on peut enlever (algo glouton)
        for u in np.argwhere(solution).flatten():
            if u > 0:
                can_remove = True

                # on regarde si les cibles captées seraient suffisemment captées sans le capteur
                list_capt = np.argwhere(self.instance.E_capt[u].flatten()).flatten()
                n_capted = len(list_capt)
                i = 0
                while (i < n_capted) and can_remove:
                    v = list_capt[i]
                    can_remove = coverage_vect[v] < 0
                    i += 1

                # ensuite on regarde si la connexité est conservée
                solution_without_u = deepcopy(solution)
                solution_without_u[u] = 0
                can_remove = can_remove and subgraph_is_connex(self.instance.E_com, np.argwhere(solution_without_u).flatten())

                # on enleve les sommets et on met à jour les sommets captés
                if can_remove:
                    solution[u] = 0
                    for target in list_capt:
                        coverage_vect[target] += 1

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
        coverage_vect += self.instance.E_capt[v].flatten()

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
        coverage_vect -= self.instance.E_capt[v].flatten()

    def is_valid(self, instance, solution_coverage=None):
        """verifie que la solution respecte bien les contraintes de l'instance
        attention il faut s'assurer que covergae_vect est valide avant l'execution

        Args:
            instance (Instance): instance à laquelle correspond la solution
            solution_coverage (tuple (solution, coverage), optional): si renseignée, evalue la validité du couple solution, coverage. valeur par defaut : None.
        """

        if solution_coverage is None:
            res = np.all(self.coverage_vect <= 0)

            if not res:
                self.valid = False
            
            else:
                self.valid = subgraph_is_connex(instance.E_com, np.argwhere(self.solution).flatten())

            return self.valid
        
        else:
            res = np.all(solution_coverage[1] <= 0)

            if not res:
                return False
            
            else:
                return subgraph_is_connex(instance.E_com, np.argwhere(solution_coverage[0]).flatten())

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

    def exchange21(self, solution, coverage_vect, v_out1, v_out2, v_in):
        """ transformation qui deplace deux capteurs de v_out et v_out et en place une en v_in
            il faut s'assurer en amont que la solution possede des capteur en v_out1 et v_out2 et pas en v_in

        Args:
            solution (np.array): vecteur de taille n_targets+1 avec des 1 aux emplacements des capteurs
            coverage_vect (np.array): vecteur de taille n_targets+1 stockant le nombre de capteurs captant chaque sommet
            v_out (int): indice de l'emplacement du capteur sortant
            v_in (int): indice de l'emplacement du capteur entrant

        modifie solution et coverage_vect
        """
        self.delete_capt(solution, coverage_vect, v_out1)
        self.delete_capt(solution, coverage_vect, v_out2)
        self.add_capt(solution, coverage_vect, v_in)

    def fitness_solution(self, solution, coverage_vect, pen_capt=2, pen_connexity=3):
        #print(np.sum(solution) - 1, pen_capt * np.linalg.norm(np.maximum(0, coverage_vect), ord=1), pen_connexity * n_connex_components(self.instance.E_com, np.argwhere(solution).flatten()))
        return np.sum(solution) - 1 + pen_capt * np.linalg.norm(np.maximum(0, coverage_vect), ord=1) + pen_connexity * (n_connex_components(self.instance.E_com, np.argwhere(solution).flatten()) -1)

    def best_in_neighbourhood(self, solution, coverage_vect, n_neighbours, p11=.5, p22=.5, p21=0, list_transfo=None):
        """trouve le meilleur voisin

        Args:
            solution (np.array 1D): solution stoque les solution avec un vecteur de 0 ou 1,
            coverage_vect (np.array 1D): vecteur stockant le nombre de fois où chaque
                            sommet est couvert
            n_neighbours (int): nombre de voisins générés par voisinage
            p11 (float, optional): probabilité de la transition échange 1-1 dans le voisinage. Defaults to .5.
            p22 (float, optional): probabilité de la transition échange 2-2 dans le voisinage. Defaults to .5.
            p21 (int, optional): probabilité de la transition échange 2-1 dans le voisinage. Defaults to 0.
            list_transfo (liste, optional): si renseigné, liste stockant les transformations. Defaults to None.

        Returns:
            solution, coverage, fitness : meilleur voisin
        """

        # p12 = 1 - p11 - p21 - p12 
        # peut bugguer si on ne peut pas choisir les sommets hors et dans l'arbre
        best_fitness = np.inf
        for i in range(n_neighbours):
            new_solution = deepcopy(solution)
            new_coverage = deepcopy(coverage_vect)

            choice = np.random.choice(np.arange(1, 4), p=[p11, p22, p21])

            if choice == 1:
                v_out = np.random.choice(np.argwhere(solution[1:]).flatten() + 1)
                v_in = np.random.choice((np.argwhere(1 - solution[1:])).flatten() + 1) 
                self.exchange11(new_solution, new_coverage, v_out, v_in)

            elif choice == 2:
                v_out = np.random.choice(np.argwhere(solution[1:]).flatten() + 1, 2, replace=False) 
                v_in = np.random.choice((np.argwhere(1 - solution[1:])).flatten() + 1, 2, replace=False)
                self.exchange22(new_solution, new_coverage, v_out[0], v_out[1], v_in[0], v_in[1])

            elif choice == 3:
                v_out = np.random.choice(np.argwhere(solution[1:]).flatten() + 1, 2, replace=False)
                v_in = np.random.choice((np.argwhere(1 - solution[1:])).flatten() + 1)
                self.exchange21(new_solution, new_coverage, v_out[0], v_out[1], v_in)

            new_fitness = self.fitness_solution(new_solution, new_coverage)
            if new_fitness < best_fitness:
                best_solution = deepcopy(new_solution)
                best_coverage = deepcopy(new_coverage)
                best_fitness = new_fitness
                if list_transfo is not None:
                    if choice == 1:
                        best_transfo = [11, v_out, v_in]

                    elif choice == 2:
                        best_transfo = [22,(v_out[0],v_out[1]),(v_in[0],v_in[1])]

                    elif choice == 3:
                        best_transfo = [21, (v_out[0],v_out[1]), v_in]

        if list_transfo is not None:
            list_transfo.append(best_transfo)
        return best_solution, best_coverage, best_fitness

    def deteriorate_solution(self, solution, coverage_vect, n_neighbours, list_transfo=None):
        """application de best_in_neighbourhood avec p11=p22=0 et p21=1"""
        return self.best_in_neighbourhood(solution, coverage_vect, n_neighbours, p11=0, p22=0, p21=1.0, list_transfo=list_transfo)

    def search(self, iter_max, time_limit, n_neighbours, stats=False):
        """recherche locale

        Args:
            iter_max (int): nombre d'itérations autorisées sans amélioration
            time_limit (float): limite de temps à la boucle
            n_neighbours (int): nombre de voisins calculés à chaque visite de voisinage
            stats (bool, optional): Booléen indiquant si on trave les graphes à la fin. Defaults to False.
        """        
        solution, coverage_vect = self.GenerateInitialSolution() # solution realisable
        self.improve_solution(solution, coverage_vect)
        n_iter = 0

        begin = time.time()

        best_solution = deepcopy(solution)
        best_coverage = deepcopy(coverage_vect)
        best_fitness = np.sum(best_solution) - 1

        if stats:
            list_transfo = []
            tab_best_solution_value = []
            tab_current_solution_fitness = []
            tab_best_neighbour_fitness = []
        else:
            list_transfo=None

        current_solution, current_coverage, current_fitness = self.deteriorate_solution(best_solution, best_coverage, n_neighbours, list_transfo=list_transfo)
    
        while self.is_valid(self.instance, solution_coverage=(current_solution, current_coverage)):
            # En enlevant un capteur on est retombé sur une solution faisable
            best_solution, best_coverage = deepcopy(current_solution), deepcopy(current_coverage)
            best_fitness = np.sum(best_solution) - 1
            current_solution, current_coverage, current_fitness = self.deteriorate_solution(best_solution, best_coverage, n_neighbours, list_transfo=list_transfo)

        iter_without_improvement = 0

        while time.time() - begin < time_limit and iter_without_improvement < iter_max:

            best_neighbour = self.best_in_neighbourhood(current_solution, current_coverage, n_neighbours, p11=.5, p22=.5, p21=0.0, list_transfo=list_transfo)
            n_iter += 1

            if best_neighbour[2] < current_fitness:
                iter_without_improvement = 0
                current_solution, current_coverage, current_fitness = best_neighbour
        
                while self.is_valid(self.instance, solution_coverage=(current_solution, current_coverage)):
                    #si on a une solution valide
                    best_solution = deepcopy(current_solution)
                    best_coverage = deepcopy(current_coverage)
                    best_fitness = np.sum(best_solution) - 1
                    current_solution, current_coverage, current_fitness = self.deteriorate_solution(best_solution, best_coverage, n_neighbours, list_transfo=list_transfo)
            else:
                iter_without_improvement += 1
                        
            if stats:       
                tab_best_neighbour_fitness.append(best_neighbour[2])
                tab_current_solution_fitness.append(current_fitness)
                tab_best_solution_value.append(best_fitness)

        self.improve_solution(solution, coverage_vect)
        self.set_solution(best_solution, best_coverage)

        if stats:
            fig1 = plt.figure(f"Local Search", figsize=(15, 10))
            ax1 = fig1.add_subplot(111)
            ax1.plot(np.arange(n_iter), tab_best_neighbour_fitness, label="Score of the best neighbour of the current solution found", c="orange", alpha=0.5)
            ax1.plot(np.arange(n_iter), tab_best_solution_value, label="Score of the best solution found so far", c="b")
            ax1.plot(np.arange(n_iter), tab_current_solution_fitness, label="Score of the current solution", c="green")
            ax1.grid(True)
            ax1.set_xlabel("Number of iterations")
            ax1.set_ylabel("value")
            ax1.set_title("Local Search : Evolution of the scores of the different solutions")
            ax1.legend()
            plt.show()

            fig2 = plt.figure(f"Local Search", figsize=(15, 10))
            ax2 = fig2.add_subplot(111)
            tab_transfos = np.zeros((3, len(list_transfo)))
            vec = [0, 0, 0]
            for i, e in enumerate(list_transfo):
                if e[0] == 11:
                    vec[0] += 1
                elif e[0] == 22:
                    vec[1] += 1
                elif e[0] == 21:
                    vec[2] += 1
                tab_transfos[:,i] = vec
            ax2.plot(range(len(list_transfo)), tab_transfos[0,:], c="orange", label="exchange 1-1")
            ax2.plot(range(len(list_transfo)), tab_transfos[1,:], c="green", label="exchange 2-2")
            ax2.plot(range(len(list_transfo)), tab_transfos[2,:], c="blue", label="exchange 1-2")
            ax2.set_xlabel("Number of transformations applied")
            ax2.set_ylabel("Aggregations of accepted transformations")
            ax2.set_title("Local Search : Count of accepted transformations")
            ax2.legend()
            ax2.grid(True)
            plt.show()

