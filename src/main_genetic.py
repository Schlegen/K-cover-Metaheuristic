from instance_class import Instance
from genetic_class import AlgoGenetic, TrivialSolutionRandomized
from chromosome_class import Chromosome
from solution_class import TrivialSolution, Solution
from utils.build_grid import extract_points_random
import time
import argparse

data_folder = "data/"
data_file = data_folder + "captANOR150_7_4.dat"
data_file = data_folder + "captANOR400_7_10_2021.dat"
data_file = data_folder + "captANOR900_14_20_2021.dat"
data_file = data_folder + "grille1010_1.dat"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="path to the instance", type=str, default="data/grille1010_1.dat")
    parser.add_argument("-rcom", "--rcom", help="value of R_com", type=int, default=1)
    parser.add_argument("-rcapt", "--rcapt", help="Value of R_capt", type=int, default=1)
    parser.add_argument("-k", "--k", help="Value of k", type=int, default=1)

    # parser.add_argument("-t", "--timelimit", help="Time limit (seconds)", type=int, default=5)
    parser.add_argument("-i", "--itermax", help="Number of iterations without improvement", type=int, default=10)
    parser.add_argument("-neighb", "--neighbours", help="Size of the neighbourhoods", type=int, default=10)

    args = parser.parse_args()

    data_file = args.data_path
    Rcom = args.rcom
    Rcapt = args.rcapt
    k = args.k
    iter_max = args.itermax
    n_neighbours = args.neighbours

    with_float = "captANOR" in data_file

    start = time.time()

    instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
    sol = AlgoGenetic(instance, nb_initial_solutions=16, nb_max_neighbours=n_neighbours,
                      proba_mutation=0.25, nb_iter_tabou=5, size_tabou=3)
    sol.run_algorithm(nb_iter=iter_max, time_limit=500)

    end = time.time()
    print(f"Computation time : {round(end - start, 2)} seconds.")

    # python main_genetic.py -d data/captANOR150_7_4.dat -rcom 1 -rcapt 1 -k 3 -i 15 -neighb 40
    # python main.py -m genetic -d data/grille1010_1.dat -rcom 1 -rcapt 1 -k 1 -i 7 -neighb 100

