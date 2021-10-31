from instance_class import Instance
from solution_class import Solution
from chromosome_class import Chromosome, TrivialSolutionRandomized
import numpy as np
import random as rd
from copy import deepcopy
import matplotlib.pyplot as plt


class AlgoGenetic:
    def __init__(self, instance, nb_initial_solutions=8):
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

            # M1
            # selected_1 = self.roulette_wheel_selection()
            # while selected_1 in selected_chromosomes:
            #     selected_1 = self.roulette_wheel_selection()
            # selected_chromosomes.append(selected_1)
            #
            # selected_2 = self.roulette_wheel_selection()
            # while selected_2 in selected_chromosomes:
            #     selected_2 = self.roulette_wheel_selection()
            # selected_chromosomes.append(selected_2)
            #
            # pairs.append((selected_1, selected_2))

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
    def mutation(self, proba=0.2):
        for i in range(len(self.children)):
            r = rd.random()
            if r < proba:
                if r < proba / 2:
                    # 1st Tabou : Neighborhood : we switch the state of 1 target (0 --> 1 or 1--> 0)
                    solution_binary, value, pen = self.children[i].tabu_search(size=8, max_iter=16)
                else:
                    # 2nd Tabou : Neighborhood : we switch the state of 2 or 3 targets
                    # (select 2 captors + 1 without captor and try permutations)
                    solution_binary, value, pen = self.children[i].tabu_search_2(size=8, max_iter=12)

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
    def evolutionary_algorithm(self, nb_iter):
        solutions_values = list()
        for iteration in range(nb_iter):
            # Init fitness values
            self.fitness_values = list()
            self.cumulative_fitness_values = list()

            # Compute the fitness function for all the population
            self.population = deepcopy(self.init_fitness_value(self.population))

            values = [self.population[i].nb_captors for i in range(len(self.population))]
            pen = [self.population[i].penalization for i in range(len(self.population))]

            print(f"\n=== [ {iteration} / {nb_iter} ] ===")
            print(values)
            print(pen)
            solutions_values.append(values)

            # Roulette wheel selection
            pairs = self.pair_selection()

            # Crossover step
            self.disk_crossover(pairs)
            for i in range(len(self.children)):
                self.children[i].penalization = 0
                if not self.children[i].is_valid(self.instance):
                    # self.children[i].reparation_heuristic(self.instance)
                    self.children[i].penalization = self.children[i].penalize_infeasibility()

            # Mutation step
            self.mutation()

            # Population update
            best_parents_indexes = np.array(self.fitness_values).argsort()[:self.nb_parents_to_keep]
            self.population = deepcopy(list(np.array(self.population)[best_parents_indexes]))

            self.fitness_values = list()
            self.children = deepcopy(self.init_fitness_value(self.children))
            best_children_indexes = np.array(self.fitness_values).argsort()[:self.nb_children_to_keep]

            self.population += list(np.array(self.children)[best_children_indexes])

            self.children = list()

        self.population = deepcopy(self.init_fitness_value(self.population))
        values = [self.population[i].nb_captors for i in range(len(self.population))]
        pen = [self.population[i].penalization for i in range(len(self.population))]
        print(f"\n=== [ {nb_iter} / {nb_iter} ] ===")
        print(values)
        print(pen)
        solutions_values.append(values)

        print(f"Mean std : {self.compute_diversity_population()}")

        # After last iteration, fix infeasible solutions AND optimize feasible ones (remove irrelevant captors)
        for i in range(len(self.population)):
            if self.population[i].penalization > 0:
                self.population[i].reparation_heuristic(self.instance)
                self.population[i].penalization = self.population[i].penalize_infeasibility()
        for i in range(len(self.population)):
            for j in range(len(self.population[i].list_captors), 0, -1):
                self.population[i].try_to_remove_captor(j - 1)

        self.population = deepcopy(self.init_fitness_value(self.population))
        values = [self.population[i].nb_captors for i in range(len(self.population))]
        pen = [self.population[i].penalization for i in range(len(self.population))]
        print(f"\n=== [ After reparation heuristic + solutions enhancement] ===")
        print(values)
        print(pen)
        solutions_values.append(values)

        min_values = np.min(np.array(solutions_values), axis=1)
        max_values = np.max(np.array(solutions_values), axis=1)
        mean_values = np.mean(np.array(solutions_values), axis=1)
        iterations = np.arange(0, len(min_values))

        plt.figure("Solution value per iteration")
        plt.step(iterations, min_values, label="Best")
        plt.step(iterations, max_values, label="Worst")
        plt.step(iterations, mean_values, label="Mean")
        plt.xlabel("Iteration")
        plt.ylabel("Nb of captors")
        plt.legend()
        plt.show()

        best_solution_index = int(np.argmin(self.fitness_values))

        self.population[best_solution_index].display(self.instance)


# Parameters to vary
# transformation in tabu : change 1 bit => change 2 bits ? (bcp plus couteux attention!)


# TEST :
# remove reparation heuristic : now we keep invalid solutions but penalize them depending of its degree of infeasibility
# faire des mutations à 2 ou 3 ou 4 d'un coup

# todo
# list hyper parameters and name them (attributes class)

