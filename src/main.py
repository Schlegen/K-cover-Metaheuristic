from instance_class import Instance
from solution_class import Solution

grid_size = (10, 10)
data_folder = "data/"
data_file = data_folder + "grille1010_2.dat"

instance = Instance.from_disk(data_file, size=grid_size, Rcom=2)
instance.display()

# grid_size = (4, 4)
# data_folder = "data/"
# data_file = data_folder + "grille44_toy.dat"
#
# instance = Instance.from_disk(data_file, size=grid_size, Rcom=2)
# instance.display()

# solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
solution = Solution([])
solution.generate_trivial_solution(instance)
print(solution.is_valid(instance))
solution.display(instance)
