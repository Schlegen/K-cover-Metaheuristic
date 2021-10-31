from instance_class import Instance
from genetic_class import AlgoGenetic, TrivialSolutionRandomized
from chromosome_class import Chromosome
from solution_class import TrivialSolution, Solution
from utils.build_grid import extract_points_random
import time

data_folder = "data/"
data_file = data_folder + "grille1010_1.dat"
data_file = data_folder + "captANOR100_7_4.dat"
data_file = data_folder + "captANOR400_7_10_2021.dat"
start = time.time()

with_float = "grille" not in data_file
instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1, k=1, with_float=with_float)

test_tabu_only = False
if test_tabu_only:
    solution1 = TrivialSolutionRandomized(instance)
    solution1.display(instance)
    # best = solution1.tabu_search(size=20, max_iter=300)[0]
    best = solution1.tabu_search_2(size=20, max_iter=100)[0]
    solution1.captors_binary = best
    solution1.update_list_captors()
    solution1.display(instance)


else:
    sol = AlgoGenetic(instance, nb_initial_solutions=32)
    sol.evolutionary_algorithm(nb_iter=15)


end = time.time()
print(f"Computation time : {round(end - start, 2)} seconds.")
exit()
#
# #instance.display()
#
# # grid_size = (4, 4)
# # data_folder = "data/"
# # data_file = data_folder + "grille44_toy.dat"
# #
# # instance = Instance.from_disk(data_file, Rcom=2)
# # instance.display()
#
# # solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
# solution1 = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3), (6, 6), (6, 7)])
# # solution1 = Solution([(1, 0), (2, 0), (3, 0), (5, 2), (5, 1), (0, 1), (0, 2), (1, 2), (1, 3)])
# # solution1 = TrivialSolution(instance)
# solution1.display(instance)
# solution1.reparation_heuristic(instance)
#
# print("")
# print(f"Solution is valid : {solution1.check_with_disk_graph(instance)}")
# print("Solution value : ", solution1.value())
# # solution1.display(instance)

# solution3 = TrivialSolutionRandomized(instance)
# # print(solution3.is_valid(instance))
# # print("Trivial Randomized Solution value : ", solution3.value())
# # solution3.display(instance)


instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1)
solution1 = TrivialSolutionRandomized(instance)
start = time.time()
print(solution1.check_with_disk_graph(instance))
end = time.time()
print(f"Computation time : {round(end - start, 2)} seconds.")