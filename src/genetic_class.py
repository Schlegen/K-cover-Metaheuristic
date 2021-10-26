from instance_class import Instance
from solution_class import Solution
from chromosome_class import Chromosome, TrivialSolutionRandomized
import numpy as np
import random as rd
from copy import deepcopy


class AlgoGenetic:
    def __init__(self, instance, nb_initial_solutions=8):
        """constructeur de la classe pour l'algo evolutionnaire
        Args:

        """
        self.instance = instance
        self.population = list()
        self.fitness_values = list()
        self.cumulative_fitness_values = list()
        self.init_population(nb_initial_solutions)
        self.init_fitness_value()

        self.children = list()
        self.nb_pairs = int(nb_initial_solutions / 4)

    def init_population(self, n):
        for k in range(n):
            solution = TrivialSolutionRandomized(self.instance)
            self.population.append(solution)

    def init_fitness_value(self):
        sum_values = 0
        for chromosome in self.population:
            chromosome.nb_captors = chromosome.value()
            self.fitness_values.append(0)
            sum_values += chromosome.nb_captors

        for i in range(len(self.population)):
            chromosome = self.population[i]
            self.fitness_values[i] = chromosome.nb_captors / sum_values

        self.cumulative_fitness_values = np.cumsum(np.array(self.fitness_values))

    # selection operator
    def roulette_wheel_selection(self):
        r = rd.random()
        greater = r < self.cumulative_fitness_values
        index = np.argmax(greater)
        return index

    def pair_selection(self):
        assert(self.nb_pairs <= len(self.population) / 2)
        pairs = list()
        selected_chromosomes = list()
        for i in range(self.nb_pairs):
            selected_1 = self.roulette_wheel_selection()
            while selected_1 in selected_chromosomes:
                selected_1 = self.roulette_wheel_selection()
            selected_chromosomes.append(selected_1)

            selected_2 = self.roulette_wheel_selection()
            while selected_2 in selected_chromosomes:
                selected_2 = self.roulette_wheel_selection()
            selected_chromosomes.append(selected_2)

            pairs.append((selected_1, selected_2))
        return pairs

    # crossover operator
    def two_points_crossover(self):
        pairs = self.pair_selection()
        for i in range(len(pairs)):
            chromosome_1 = deepcopy(self.population[pairs[i][0]])
            chromosome_2 = deepcopy(self.population[pairs[i][1]])

            n = len(chromosome_1.captors_binary)
            index_for_exchange = rd.randint(0, n)  # after/before this index, we exchange
            side_for_exchange = rd.randint(0, 1)  # 0 : change left of the vector, 1 : right
            indexes_for_exchange = [j for j in range(index_for_exchange)] if side_for_exchange == 0 \
                else [j for j in range(index_for_exchange, n)]

            print("Before exchange")
            print(chromosome_1.captors_binary)
            print(chromosome_2.captors_binary)

            copy_chromosome_1 = deepcopy(chromosome_1.captors_binary)
            for j in indexes_for_exchange:
                chromosome_1.captors_binary[j] = chromosome_2.captors_binary[j]
                chromosome_2.captors_binary[j] = copy_chromosome_1[j]

            print("After exchange")
            print(chromosome_1.captors_binary)
            print(chromosome_2.captors_binary)

    # mutation operator
    def mutation(self):
        pass
