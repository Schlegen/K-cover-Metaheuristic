import unittest
from instance_class import Instance
from solution_class import Solution

class BatteryTests(unittest.TestCase):

    ## fonction is_valid de la classe solution
    def test_solution_is_valid1(self):
        instance = Instance.from_disk("data/grille44_toy.dat", size=(4,4), Rcapt=1, Rcom=1)

        solution = Solution([(1,0), (2,0), (3,0), (3,1), (3,2), (3,3), (0,1), (0,2) ,(1,2), (1,3)])

        self.assertTrue(solution.is_valid(instance))

    def test_solution_is_valid2(self):
        instance = Instance.from_disk("data/grille44_toy.dat", size=(4,4), Rcapt=1, Rcom=1)

        solution = Solution([(1,0), (2,0), (3,0), (3,2), (3,3), (0,1), (0,2) ,(1,2), (1,3)])

        self.assertFalse(solution.is_valid(instance))

    def test_solution_is_valid2(self):
        instance = Instance.from_disk("data/grille44_toy.dat", size=(4,4), Rcapt=1, Rcom=2)

        solution = Solution([(1,0), (2,0), (3,0), (3,2), (3,3), (0,1), (0,2) ,(1,2), (1,3)])

        self.assertTrue(solution.is_valid(instance))

        def test_solution_is_valid2(self):
        instance = Instance.from_disk("data/grille44_toy.dat", size=(4,4), Rcapt=2, Rcom=2)

        solution = Solution([(1,0), (3,0), (3,2), (3,3), (0,1), (0,2) ,(1,2), (1,3)])

        self.assertTrue(solution.is_valid(instance))


if __name__ == '__main__':
    unittest.main()