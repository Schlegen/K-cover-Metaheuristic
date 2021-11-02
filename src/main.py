from instance_class import Instance
from solution_class import Solution, LocalSearch
from genetic_class import AlgoGenetic
from utils.errors import InputError
import pandas as pd
import argparse
import time
import numpy as np
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of the execution",
                    choices=["bound", "genetic", "localsearch", "tabu", "test"], default="test")
    parser.add_argument("-d", "--data_path", help="path to the instance", type=str, default="data/grille1010_1.dat")
    parser.add_argument("-s", "--save_file", help="Path to the score file", type=str, default="data/scores.csv")
    parser.add_argument("-rcom", "--rcom", help="value of R_com", type=int, default=1)
    parser.add_argument("-rcapt", "--rcapt", help="Value of R_capt", type=int, default=1)
    parser.add_argument("-k", "--k", help="Value of k", type=int, default=1)

    #mode relaxation
    parser.add_argument("-cplex", "--cplex", help="Path to the cplex executable", type=str, default="")

    #mode localsearch
    parser.add_argument("-t", "--timelimit", help="Time limit (seconds)", type=int, default=200)
    parser.add_argument("-i", "--itermax", help="Number of iterations without improvement", type=int, default=10)
    parser.add_argument("-neighb", "--neighbours", help="Size of the neighbourhoods", type=int, default=100)

    #mode d'execution 
    parser.add_argument("--stats", action='store_true', help="Mode to get plots and stats about methods")
    parser.add_argument("--optim", action='store_true', help="Mode used to compare methods")
    parser.set_defaults(stats=False)
    parser.set_defaults(optim=True)
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
    stats = args.stats
    optim = args.optim
    with_float = "captANOR" in data_file

    start = time.time()

    if mode == "bound":
        from milp_class import MilpApproach

        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom)
        milp = MilpApproach(instance)
        if path_cplex != "":
            milp.relaxation_value(path_cplex)

            if os.path.isfile(save_file):
                df = pd.read_csv(save_file, sep=";")

            else:
                raise InputError("Pas de fichier de score trouvé")
                #df = pd.DataFrame(columns = ["file", "Rcapt", "Rcom", "lower_bound"])
            
            df = df.append({"file" : data_file, "Rcapt" : Rcapt, "Rcom" : Rcom, "lower_bound" : milp.relaxation_value}, ignore_index=True)
            df.to_csv(save_file, sep=";", index=False)
        else:
            raise InputError("Pas d'exécutable CPLEX renseigné")

    elif mode == "genetic":
        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        if optim:
            list_values = []
            list_times = []
            launch = 0

            while time.time() - start < time_limit and launch < 10:
                launch += 1
                local_search = LocalSearch(instance)
                remaining_time = start + time_limit - time.time()
                sol = AlgoGenetic(instance, nb_initial_solutions=32, nb_max_neighbours=n_neighbours, proba_mutation=0.2)
                final_value = sol.run_algorithm(nb_iter=iter_max, time_limit=remaining_time, show_final_solution=stats)

                list_values.append(sol.value)
                list_times.append(remaining_time - (start + time_limit - time.time()))
                
            list_value = np.array(list_values)
            list_times = np.array(list_times)
            avg_val = np.mean(list_value)
            std_val = np.std(list_value)
            min_val = np.min(list_value)
            avg_time = np.mean(list_times)

            df = pd.read_csv(save_file, sep=";")
            df = df.set_index(["Instance", "Rcapt", "Rcom", "k"])

            index = (data_file, Rcapt, Rcom, k)

            if df.index.isin([index]).any():
                df.at[index, "number of launchs genetic"] = launch
                df.at[index, "min value genetic"] = min_val
                df.at[index, "mean value genetic"] = round(avg_val, 2)
                df.at[index, "std value genetic"] = round(std_val, 2)
                df.at[index, "mean CPU time genetic"] = round(avg_time, 2)
                df.reset_index(inplace=True)


            else:
                df.reset_index(inplace=True)
                df = df.append({"Instance" : data_file, "Rcapt" : Rcapt, "Rcom" : Rcom, "k" : k,
                    "mean CPU time genetic": round(avg_time, 2), "min value genetic" : min_val,
                    "mean value genetic" : round(avg_val, 2), "std value genetic" : round(std_val, 2),
                    "number of launchs genetic" : launch},
                    ignore_index=True)

            df.to_csv(save_file, sep=";", index=False)



        else:
            sol = AlgoGenetic(instance, nb_initial_solutions=32, nb_max_neighbours=n_neighbours, proba_mutation=0.2)
            final_value = sol.run_algorithm(nb_iter=iter_max, time_limit=time_limit, show_final_solution=stats)
            end = time.time()
            computation_time = round(end - start, 3)

    elif mode == "localsearch":
        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom, k=k, with_float=with_float)
        if optim:
            list_values = []
            list_times = []
            launch = 0

            while time.time() - start < time_limit and launch < 20:
                launch += 1
                local_search = LocalSearch(instance)
                remaining_time = start + time_limit - time.time()
                local_search.search(iter_max, remaining_time, n_neighbours, stats=stats)
                list_values.append(local_search.value())
                list_times.append(remaining_time - (start + time_limit - time.time()))
                
            list_value = np.array(list_values)
            list_times = np.array(list_times)
            avg_val = np.mean(list_value)
            std_val = np.std(list_value)
            min_val = np.min(list_value)
            avg_time = np.mean(list_times)

            df = pd.read_csv(save_file, sep=";")
            df = df.set_index(["Instance", "Rcapt", "Rcom", "k"])

            index = (data_file, Rcapt, Rcom, k)

            if df.index.isin([index]).any():
                df.at[index, "number of launchs localsearch"] = launch
                df.at[index, "min value localsearch"] = min_val
                df.at[index, "mean value localsearch"] = round(avg_val, 2)
                df.at[index, "std value localsearch"] = round(std_val, 2)
                df.at[index, "mean CPU time localsearch"] = round(avg_time, 2)
                df.reset_index(inplace=True)

            else:
                df.reset_index(inplace=True)
                df = df.append({"Instance" : data_file, "Rcapt" : Rcapt, "Rcom" : Rcom, "k" : k,
                    "number of launchs localsearch"	: launch,
                  "min value localsearch" : min_val, "mean value localsearch" : round(avg_val, 2), "std value localsearch" : round(std_val, 2),
                  "mean CPU time localsearch" : round(avg_time,2)}, ignore_index=True)
            df.to_csv(save_file, sep=";", index=False)

        else: #mode classique
            local_search = LocalSearch(instance)
            local_search.search(iter_max, time_limit, n_neighbours, stats=stats)
            print("solution après recherche : ", local_search.value())
            print("validité", local_search.is_valid(instance))
            local_search.display(instance)

    elif mode == "test":
        # Testing mode (for debug)
        data_file = "data/grille1010_1.dat"
        instance = Instance.from_disk(data_file, Rcom=2, Rcapt=1, k=1, with_float=False)
        sol = Solution([(0, 1), (0, 5), (0, 6), (0, 8), (1, 3), (1, 5), (1, 6), (1, 9), (2, 0), (2, 2), (2, 4), (2, 8), (3, 4), (4, 0), (4, 1), (5, 1), (5, 3), (5, 8), (5, 9), (6, 1), (6, 3), (6, 4), (6, 5), (7, 1), (7, 7), (7, 9), (8, 0), (8, 1), (8, 5), (9, 5), (9, 6), (9, 8), (3, 9)])
        print(f"Solution is valid : {sol.is_valid(instance)}")
        sol.display(instance)