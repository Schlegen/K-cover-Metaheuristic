from instance_class import Instance
from genetic_class import AlgoGenetic, TrivialSolutionRandomized
from chromosome_class import Chromosome
from solution_class import TrivialSolution, Solution
from utils.build_grid import extract_points_random
import time

data_folder = "data/"
data_file = data_folder + "captANOR150_7_4.dat"
data_file = data_folder + "grille1010_1.dat"
start = time.time()

instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1, k=1, with_float=False)
sol = Solution([(0, 1), (0, 5), (0, 6), (0, 8), (1, 3), (1, 5), (1, 6), (1, 9), (2, 0), (2, 2), (2, 4), (2, 8), (3, 4), (4, 0), (4, 1), (5, 1), (5, 3), (5, 8), (5, 9), (6, 1), (6, 3), (6, 4), (6, 5), (7, 1), (7, 7), (7, 9), (8, 0), (8, 1), (8, 5), (9, 5), (9, 6), (9, 8), (3, 9)])
print(sol.is_valid(instance))
print(sol.display(instance))
print(sol.check_with_disk_graph(instance))
exit()
test_tabu_only = False
if test_tabu_only:
    solution1 = TrivialSolutionRandomized(instance)
    solution1.display(instance)
    # best = solution1.tabu_search(size=20, max_iter=300)[0]
    best = solution1.tabu_search_2(size=20, max_iter=100)[0]
    solution1.captors_binary = best
    solution1.update_list_captors()
    solution1.display(instance)
    # CHECK QUE DES FOIS CA AMELIORE VRMT
    # SINON, idee : au lieu de faire des changements 1 par 1, faire 2 par 2

else:
    sol = AlgoGenetic(instance, nb_initial_solutions=12)
    sol.evolutionary_algorithm(nb_iter=3)


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