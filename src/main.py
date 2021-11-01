from instance_class import Instance
from solution_class import Solution, TrivialSolution, TrivialSolutionRandomized0, LocalSearch
from genetic_class import AlgoGenetic
import time
from utils.errors import InputError
from milp_class import MilpApproach
import argparse
import pandas as pd
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of the execution",
                    choices=["bound", "genetic", "local", "tabu", "test"], default="test")
    parser.add_argument("-d", "--data_path", help="path to the instance", type=str, default="data/grille1010_1.dat")
    parser.add_argument("-s", "--save_file", help="Path to the score file", type=str, default="data/scores.csv")
    parser.add_argument("-rcom", "--rcom", help="value of R_com", type=int, default=1)
    parser.add_argument("-rcapt", "--rcapt", help="Value of R_capt", type=int, default=1)
    parser.add_argument("-k", "--k", help="Value of k", type=int, default=1)

    #mode relaxation
    parser.add_argument("-cplex", "--cplex", help="Path to the cplex executable", type=str, default="")

    #mode localsearch
    parser.add_argument("-t", "--timelimit", help="Time limit (seconds)", type=int, default=5)
    parser.add_argument("-i", "--itermax", help="Number of iterations without improvement", type=int, default=10)
    parser.add_argument("-neighb", "--neighbours", help="Size of the neighbourhoods", type=int, default=40)

    #mode d'execution 
    parser.add_argument("--stats", help="Time limit (seconds)", type=int, default=5)

    args = parser.parse_args()

    data_file = args.data_path
    save_file = args.save_file
    mode = args.mode
    Rcom = args.rcom
    Rcapt = args.rcapt
    k = args.k
    time_limit = args.timelimit
    iter_max = args.itermax
    path_cplex = args.cplex
    n_neighbours = args.neighbours

    with_float = "captANOR" in data_file

    if mode == "bound":
        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom)
        milp = MilpApproach(instance)
        if path_cplex != "":
            milp.relaxation_value(path_cplex)

            if os.path.isfile(save_file):
                df = pd.read_csv(save_file, sep=";")

            else:
                df = pd.DataFrame(columns = ["file", "Rcapt", "Rcom", "lower_bound"])
            
            df = df.append({"file" : data_file, "Rcapt" : Rcapt, "Rcom" : Rcom, "lower_bound" : milp.relaxation_value}, ignore_index=True)
            df.to_csv(save_file, sep=";", index=False)
        else:
            raise InputError("Pas d'exécutable CPLEX renseigné")

    elif mode == "genetic":
        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        sol = AlgoGenetic(instance, nb_initial_solutions=32, nb_max_neighbours=n_neighbours, proba_mutation=0.2)
        sol.evolutionary_algorithm(nb_iter=15)

    elif mode == "local":

        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        local_search = LocalSearch(instance)
        #local_search.set_solution(*local_search.GenerateInitialSolution())
        # print("2")
        # #local_search.coverage_as_matrix(local_search.coverage_vect)
        # # solution, coverage = local_search.GenerateInitialSolution()
        # #local_search.improve_solution(solution, coverage)
        # #local_search.set_solution(solution, coverage)
        # local_search.is_valid(instance)
        # local_search.display(instance)



        # # print("solution initiale : ", local_search.value())
        local_search.search(iter_max, time_limit, n_neighbours)
        #print("solution après amélioration : ", local_search.value())
        local_search.is_valid(instance)
        local_search.display(instance)

    elif mode == "test":
        data_file = "data/grille1010_1.dat"
        start = time.time()

        instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1, k=1, with_float=False)
        sol = Solution([(0, 1), (0, 5), (0, 6), (0, 8), (1, 3), (1, 5), (1, 6), (1, 9), (2, 0), (2, 2), (2, 4), (2, 8), (3, 4), (4, 0), (4, 1), (5, 1), (5, 3), (5, 8), (5, 9), (6, 1), (6, 3), (6, 4), (6, 5), (7, 1), (7, 7), (7, 9), (8, 0), (8, 1), (8, 5), (9, 5), (9, 6), (9, 8), (3, 9)])
        print(sol.is_valid(instance))
        print(sol.display(instance))
        # Partie ou on mets des trucs temporaires pour les tester
