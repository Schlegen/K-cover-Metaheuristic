from instance_class import Instance
from solution_class import Solution, TrivialSolution, MinCostFlowMethod, TrivialSolutionRandomized
import time

data_folder = "data/"
data_file = data_folder + "grille2020_1.dat"

instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1)
# instance.display()

start = time.time()
# solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
# solution1 = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3), (6, 6), (6, 7)])
# solution1 = TrivialSolution(instance)
solution1 = TrivialSolutionRandomized(instance)
print(f"Solution is valid : {solution1.is_valid(instance)}")
print("Solution value :", solution1.value())
end = time.time()
print(f"Computation time : {round(end - start, 2)} seconds.")
solution1.display(instance)

# solution3 = TrivialSolutionRandomized(instance)


# solution2 = MinCostFlowMethod(instance)
# solution2.compute_flow()
# solution2.build_solution()
# print("comparaison : ", solution1.value(), solution2.value())
