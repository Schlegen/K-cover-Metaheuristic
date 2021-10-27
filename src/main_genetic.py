from instance_class import Instance
from genetic_class import AlgoGenetic

data_folder = "data/"
data_file = data_folder + "grille1010_1.dat"

instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1)

sol = AlgoGenetic(instance, nb_initial_solutions=16)
sol.evolutionary_algorithm(nb_iter=40)
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
#
# solution3 = TrivialSolutionRandomized(instance)
# # print(solution3.is_valid(instance))
# # print("Trivial Randomized Solution value : ", solution3.value())
# # solution3.display(instance)
#
#
# # solution2 = MinCostFlowMethod(instance)
# # solution2.compute_flow()
# # solution2.build_solution()
# # print("comparaison : ", solution1.value(), solution2.value())
