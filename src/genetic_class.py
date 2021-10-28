from instance_class import Instance
from solution_class import Solution
from chromosome_class import Chromosome, TrivialSolutionRandomized
import numpy as np
import random as rd
from copy import deepcopy
import matplotlib.pyplot as plt


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

        self.children = list()
        self.nb_pairs = int(nb_initial_solutions / 2)

        self.nb_parents_to_keep = int(nb_initial_solutions / 2) + 1
        self.nb_children_to_keep = nb_initial_solutions - self.nb_parents_to_keep

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
            sum_values += population[i].nb_captors

        for i in range(n):
            chromosome = population[i]
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
    def two_points_crossover(self, pairs):
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
        radius_for_crossover = rd.randint(self.instance.Rcom, 3 * self.instance.Rcom)
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
    def mutation(self, proba=0.1):
        for i in range(len(self.children)):
            print(self.children[i].captors_binary)
            r = rd.random
            if r < proba:
                # TODO : mutation
                # on fait un tabou au sein d'un disque random avec pénalisation si irréalisable (plutot que de rendre
                # realisable c'est plus simple et de toute ca degage juste derriere)

                print("Mutation done")

                self.children[i].update_list_captors()
        pass

    # main function
    def evolutionary_algorithm(self, nb_iter):
        solutions_values = list()
        for iteration in range(nb_iter):
            # Init fitness values
            self.fitness_values = list()
            self.cumulative_fitness_values = list()

            # Compute the fitness function for all the population
            self.init_fitness_value(self.population)

            print("")
            print(f"Values at iteration {iteration}")
            values = [self.population[i].nb_captors for i in range(len(self.population))]
            print(values)
            solutions_values.append(values)

            # Roulette wheel selection
            pairs = self.pair_selection()

            # Crossover step
            self.disk_crossover(pairs)
            for i in range(len(self.children)):
                if not self.children[i].is_valid(self.instance):
                    self.children[i].reparation_heuristic(self.instance)

            # Mutation step
            # self.mutation()
            # for i in range(len(self.children)):
            #     if not self.children[i].is_valid():
            #         self.children[i].reparation_heuristic(self.instance)

            # Population update
            best_parents_indexes = np.array(self.fitness_values).argsort()[:self.nb_parents_to_keep]
            self.population = deepcopy(list(np.array(self.population)[best_parents_indexes]))

            # M2 : try to select best children with roulette ?
            self.fitness_values = list()
            self.init_fitness_value(self.children)
            best_children_indexes = np.array(self.fitness_values).argsort()[:self.nb_children_to_keep]
            self.population += list(np.array(self.children)[best_children_indexes])

            self.children = list()

            print(self.population[0].captors_binary)
            print(self.population[1].captors_binary)

        print("\nEND")
        self.init_fitness_value(self.population)
        values = [self.population[i].nb_captors for i in range(len(self.population))]
        print(values)
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

