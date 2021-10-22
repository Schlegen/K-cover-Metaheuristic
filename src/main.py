from instance_class import Instance
from solution_class import Solution, TrivialSolution, MinCostFlowMethod, TrivialSolutionRandomized

data_folder = "data/"
data_file = data_folder + "grille1010_1.dat"

instance = Instance.from_disk(data_file, Rcom=2)
#instance.display()

# grid_size = (4, 4)
# data_folder = "data/"
# data_file = data_folder + "grille44_toy.dat"
#
# instance = Instance.from_disk(data_file, Rcom=2)
# instance.display()

# solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
solution1 = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3), (6, 6), (6, 7)])
solution1 = TrivialSolution(instance)
# solution1.find_connected_components(instance)
print(solution1.is_valid(instance))
print("Trivial Solution value : ", solution1.value())
solution1.display(instance)

# solution3 = TrivialSolutionRandomized(instance)
# print(solution3.is_valid(instance))
# print("Trivial Randomized Solution value : ", solution3.value())
# solution3.display(instance)


# solution2 = MinCostFlowMethod(instance)
# solution2.compute_flow()
# solution2.build_solution()
# print("comparaison : ", solution1.value(), solution2.value())
