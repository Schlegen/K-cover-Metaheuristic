from instance_class import Instance
from solution_class import Solution, TrivialSolution, TrivialSolutionRandomized0, LocalSearch
import time
from utils.errors import InputError
from bounds import MilpApproach
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
    parser.add_argument("--neighbours", help="Size of the neighbourhoods", type=int, default=40)

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

            else :
                df = pd.DataFrame(columns = ["file", "Rcapt", "Rcom", "lower_bound"])
            
            df = df.append({"file" : data_file, "Rcapt" : Rcapt, "Rcom" : Rcom, "lower_bound" : milp.relaxation_value}, ignore_index=True)
            df.to_csv(save_file, sep=";", index=False)
        else:
            raise InputError("Pas d'exécutable CPLEX renseigné")

    elif mode == "genetic":
        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom)
    
    elif mode == "local":

        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        local_search = LocalSearch(instance)
        local_search.set_solution(*local_search.GenerateInitialSolution())
        # print("2")
        # #local_search.coverage_as_matrix(local_search.coverage_vect)
        # # solution, coverage = local_search.GenerateInitialSolution()
        # #local_search.improve_solution(solution, coverage)
        # #local_search.set_solution(solution, coverage)
        # local_search.is_valid(instance)
        # local_search.display(instance)



        # # print("solution initiale : ", local_search.value())
        #local_search.search(iter_max, time_limit, n_neighbours)
        #print("solution après amélioration : ", local_search.value())
        local_search.is_valid(instance)
        local_search.display(instance)

    elif mode == "test":
        import numpy as np

        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        local_search = LocalSearch(instance)
        solution, coverage = local_search.GenerateInitialSolution()
        local_search.set_solution(solution, coverage)
        local_search.coverage_as_matrix(coverage)
        # local_search.is_valid(instance)
        # local_search.display(instance)
        v_out = np.random.choice(np.argwhere(solution[1:]).flatten() + 1, 2, replace=False)
        v_in = np.random.choice((np.argwhere(1 - solution[1:])).flatten() + 1)
        transfo_name = f"t21_{v_out[0]},{v_out[1]}_{v_in}"
        local_search.exchange21(solution, coverage, v_out[0], v_out[1], v_in)
        print(transfo_name, instance.reversed_indexes[v_out[0]], instance.reversed_indexes[v_out[1]], instance.reversed_indexes[v_in])
        local_search.set_solution(solution, coverage)
        local_search.coverage_as_matrix(coverage)
        local_search.is_valid(instance)
        local_search.display(instance)
        # Partie ou on mets des trucs temporaires pour les tester



#    print(milp.captors)
#instance.display()

# grid_size = (4, 4)
# data_folder = "data/"
# data_file = data_folder + "grille44_toy.dat"
#
# instance = Instance.from_disk(data_file, Rcom=2)


#######################
# data_folder = "data/"
# data_file = data_folder + "grille2020_1.dat"
#
# instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1)
# # instance.display()
#
# start = time.time()
# # solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
# # solution1 = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3), (6, 6), (6, 7)])
# # solution1 = TrivialSolution(instance)
# # # solution1.find_connected_components(instance)
# # print(solution1.is_valid(instance))
# # print("Trivial Solution value : ", solution1.value())
# # solution1.display(instance)
#
# # solution3 = TrivialSolutionRandomized(instance)
# # print(solution3.is_valid(instance))
# # print("Trivial Randomized Solution value : ", solution3.value())
# # solution3.display(instance)
# solution1 = TrivialSolutionRandomized0(instance)
# print(f"Solution is valid : {solution1.is_valid(instance)}")
# print("Solution value :", solution1.value())
# end = time.time()
# print(f"Computation time : {round(end - start, 2)} seconds.")
# solution1.display(instance)
#
# # solution3 = TrivialSolutionRandomized(instance)
#
#
# # solution2 = MinCostFlowMethod(instance)
# # solution2.compute_flow()
# # solution2.build_solution()
# # print("comparaison : ", solution1.value(), solution2.value())
