from instance_class import Instance
from chromosome_class import TrivialSolutionRandomized
import numpy as np
import random as rd
from copy import deepcopy
import matplotlib.pyplot as plt
import time


class AlgoGenetic:
    def __init__(self, instance, nb_initial_solutions, nb_max_neighbours=8,
                 proba_mutation=0.2, size_tabou=6, nb_iter_tabou=12):
        """Class for Evolutionary Algorithm Method
        Args:
            instance (Instance) : instance we want to optimize
            nb_initial_solutions (int) : size of the initial population
        """
        self.instance = instance  # Instance to solve (targets coordinates, k, Rcom, Rcapt, ...)
        self.population = list()  # Population at each iteration : List of chromosomes
        self.init_population(nb_initial_solutions)  # Initialize the population with an Heuristic

        self.fitness_values = list()  # For pairs selection AND population update
        self.cumulative_fitness_values = list()  # For Roulette Wheel Selection

        self.children = list()  # Children at each iteration : List of chromosomes
        self.nb_pairs = int(nb_initial_solutions / 2)  # Nb of pairs of parents chosen at each iteration (for crossover)

        self.nb_parents_to_keep = int(nb_initial_solutions / 4) + 1  # Nb of parents we keep at population update
        self.nb_children_to_keep = nb_initial_solutions - self.nb_parents_to_keep  # Nb of children we keep

        self.nb_max_neighbours = nb_max_neighbours  # Maximum size of neighborhood for mutation step
        self.proba_mutation = proba_mutation  # Probability for a chromosome to be muted at each iteration
        self.size_tabou = size_tabou  # Size of the tabou list for the mutation step
        self.nb_iter_tabou = nb_iter_tabou  # Nb of iterations for the tabou search during the mutation step

    def init_population(self, n):
        for k in range(n):
            solution = TrivialSolutionRandomized(self.instance)
            self.population.append(solution)

    def init_fitness_value(self, population):
        n = len(population)
        self.fitness_values = [0 for i in range(n)]

        sum_values = 0
        for i in range(n):
            population[i].nb_captors = population[i].value()
            population[i].nb_captors += population[i].penalization
            sum_values += population[i].nb_captors

        for i in range(n):
            chromosome = population[i]
            self.fitness_values[i] = chromosome.nb_captors / sum_values

        # Small values => good solution => bigger probability to be chosen
        probabilities = (1 - np.array(self.fitness_values))
        probabilities = probabilities / np.sum(probabilities)
        self.cumulative_fitness_values = np.cumsum(probabilities)

        return population

    # selection operator
    def roulette_wheel_selection(self):
        r = rd.random()
        greater = r < self.cumulative_fitness_values
        index = np.argmax(greater)
        return index

    def pair_selection(self):
        assert(self.nb_pairs <= len(self.population) / 2)
        pairs = list()
        for i in range(self.nb_pairs):
            selected_1 = self.roulette_wheel_selection()
            selected_2 = self.roulette_wheel_selection()
            while selected_2 == selected_1:
                selected_2 = self.roulette_wheel_selection()
            pairs.append((selected_1, selected_2))
        return pairs

    # crossover operator
    def one_point_crossover(self, pairs):
        for i in range(len(pairs)):
            chromosome_1 = deepcopy(self.population[pairs[i][0]])
            chromosome_2 = deepcopy(self.population[pairs[i][1]])

            n = len(chromosome_1.captors_binary)
            index_for_crossover = rd.randint(0, n)  # after/before this index, we exchange
            side_for_crossover = rd.randint(0, 1)  # 0 : change left of the vector, 1 : right
            indexes_for_crossover = [j for j in range(index_for_crossover)] if side_for_crossover == 0 \
                else [j for j in range(index_for_crossover, n)]

            copy_chromosome_1 = deepcopy(chromosome_1.captors_binary)
            for j in indexes_for_crossover:
                chromosome_1.captors_binary[j] = chromosome_2.captors_binary[j]
                chromosome_2.captors_binary[j] = copy_chromosome_1[j]

            chromosome_1.update_list_captors()
            chromosome_2.update_list_captors()

            self.children.append(chromosome_1)
            self.children.append(chromosome_2)

    # crossover operator
    def disk_crossover(self, pairs):
        # we take a disk with : Rcom <= radius <= 2*Rcom
        radius_for_crossover = rd.randint(self.instance.Rcom, 2 * self.instance.Rcom)
        neighbours = self.instance.neighbours_dict(radius_for_crossover)

        for i in range(len(pairs)):
            chromosome_1 = deepcopy(self.population[pairs[i][0]])
            chromosome_2 = deepcopy(self.population[pairs[i][1]])

            n = len(chromosome_1.captors_binary)

            # Crossover for all targets within a randomly chosen disk
            index_for_center = rd.randint(0, n - 1)  # within a circle around this point, we exchange
            point_for_center = (self.instance.targets + [(0, 0)])[index_for_center]  # coordinate of this center
            points_for_crossover = [points for points
                                    in neighbours[point_for_center]]
            indexes_for_crossover = [(self.instance.targets + [(0, 0)]).index(u) for u in points_for_crossover]

            copy_chromosome_1 = deepcopy(chromosome_1.captors_binary)
            for j in indexes_for_crossover:
                chromosome_1.captors_binary[j] = chromosome_2.captors_binary[j]
                chromosome_2.captors_binary[j] = copy_chromosome_1[j]

            chromosome_1.update_list_captors()
            chromosome_2.update_list_captors()

            self.children.append(chromosome_1)
            self.children.append(chromosome_2)

    # mutation operator
    def mutation(self):
        for i in range(len(self.children)):
            r = rd.random()
            if r < self.proba_mutation:
                if r < self.proba_mutation / 2:
                    # 1st Tabou : Neighborhood : we switch the state of 1 target (0 --> 1 or 1--> 0)
                    solution_binary, value, pen = self.children[i].tabu_search(size=self.size_tabou,
                                                                               max_iter=self.nb_iter_tabou,
                                                                               nb_neighbours=self.nb_max_neighbours)
                else:
                    # 2nd Tabou : Neighborhood : we switch the state of 2 or 3 targets
                    # (select 2 captors + 1 without captor and try permutations)
                    solution_binary, value, pen = self.children[i].tabu_search_2(size=self.size_tabou,
                                                                                 max_iter=self.nb_iter_tabou,
                                                                                 nb_neighbours=self.nb_max_neighbours)

                self.children[i].captors_binary = deepcopy(solution_binary)
                self.children[i].update_list_captors()
                self.children[i].penalization = pen

    def compute_diversity_population(self):
        """ Compute the mean of the standard deviation between each solution of the population.
            Used to detect if the population has stabilized around an almost fixed solution"""
        solutions = np.array([self.population[i].captors_binary for i in range(len(self.population))])
        standard_deviation_mean = np.mean(np.std(solutions, axis=0))
        return standard_deviation_mean

    # main function
    def run_algorithm(self, nb_iter, time_limit=900, show_final_solution=True):
        solutions_values = list()
        start = time.time()
        iteration = 0
        mean_std = 1
        while iteration < nb_iter and time.time() - start < time_limit and mean_std > 0.05:
            # Init fitness values
            self.fitness_values = list()
            self.cumulative_fitness_values = list()

            # Compute the fitness function for all the population
            self.population = deepcopy(self.init_fitness_value(self.population))

            values = [self.population[i].nb_captors for i in range(len(self.population))]
            pen = [self.population[i].penalization for i in range(len(self.population))]

            print(f"\n=== [ {iteration} / {nb_iter} ] ===")
            print(f"Val. : {values}")
            print(f"Pen. : {pen}")
            solutions_values.append(values)

            # Roulette wheel selection
            pairs = self.pair_selection()

            st = time.time()
            # Crossover step
            self.disk_crossover(pairs)
            for i in range(len(self.children)):
                self.children[i].penalization = 0
                if not self.children[i].is_valid(self.instance):
                    # self.children[i].reparation_heuristic(self.instance)
                    self.children[i].penalization = self.children[i].penalize_infeasibility()
            en = time.time()
            print(f"Crossover : {round(en - st, 3)} seconds")

            # Mutation step
            st = time.time()
            self.mutation()
            en = time.time()
            print(f"Mutation : {round(en - st, 3)} seconds")

            # Population update
            best_parents_indexes = np.array(self.fitness_values).argsort()[:self.nb_parents_to_keep]
            self.population = deepcopy(list(np.array(self.population)[best_parents_indexes]))

            self.fitness_values = list()
            self.children = deepcopy(self.init_fitness_value(self.children))
            best_children_indexes = np.array(self.fitness_values).argsort()[:self.nb_children_to_keep]

            self.population += list(np.array(self.children)[best_children_indexes])

            self.children = list()

            iteration += 1
            mean_std = self.compute_diversity_population()

        self.population = deepcopy(self.init_fitness_value(self.population))
        values = [self.population[i].nb_captors for i in range(len(self.population))]
        pen = [self.population[i].penalization for i in range(len(self.population))]
        print(f"\n=== [ {nb_iter} / {nb_iter} ] ===")
        print(f"Val. : {values}")
        print(f"Pen. : {pen}")
        solutions_values.append(values)

        print(f"\nMean std within the final population: {round(self.compute_diversity_population(), 5)}")

        # After last iteration, fix infeasible solutions AND optimize feasible ones (remove irrelevant captors)
        for i in range(len(self.population)):
            if self.population[i].penalization > 0:
                self.population[i].reparation_heuristic(self.instance)

                # We check the solution is now valid (penalization = 0)
                self.population[i].penalization = self.population[i].penalize_infeasibility()

        for i in range(len(self.population)):
            for j in range(len(self.population[i].list_captors), 0, -1):
                self.population[i].try_to_remove_captor(j - 1)

        self.population = deepcopy(self.init_fitness_value(self.population))
        values = [self.population[i].nb_captors for i in range(len(self.population))]
        pen = [self.population[i].penalization for i in range(len(self.population))]
        print(f"\n=== [ After reparation heuristic + solutions enhancement] ===")
        print(f"Val. : {values}")
        print(f"Pen. : {pen}")
        solutions_values.append(values)

        min_values = np.min(np.array(solutions_values), axis=1)
        max_values = np.max(np.array(solutions_values), axis=1)
        mean_values = np.mean(np.array(solutions_values), axis=1)
        iterations = np.arange(0, len(min_values))


        if show_final_solution:
            plt.figure("Population's values per iteration")
            plt.step(iterations, min_values, label="Best")
            plt.step(iterations, max_values, label="Worst")
            plt.step(iterations, mean_values, label="Mean")
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

        best_solution_index = int(np.argmin(self.fitness_values))

        self.value = self.population[best_solution_index].value()
        print(f"\nBest Solution value : {self.value}")
        if show_final_solution:
            self.population[best_solution_index].display(self.instance)
        

        return self.population[best_solution_index].value()
