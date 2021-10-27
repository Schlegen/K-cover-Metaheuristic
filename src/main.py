from instance_class import Instance
from solution_class import Solution, TrivialSolution, TrivialSolutionRandomized, TabuSearch
from utils.errors import InputError
from bounds import MilpApproach
import argparse
import pandas as pd
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of the execution",
                    choices=["bound", "genetic", "tabu", "test"], default="test")
    parser.add_argument("-d", "--data_path", help="path to the instance", type=str, default="data/grille1010_1.dat")
    parser.add_argument("-s", "--save_file", help="Path to the score file", type=str, default="data/scores.csv")
    parser.add_argument("-rcom", "--rcom", help="value of R_com", type=int, default=1)
    parser.add_argument("-rcapt", "--rcapt", help="Value of R_capt", type=int, default=1)
    parser.add_argument("-cplex", "--cplex", help="Path to the cplex executable", type=str, default="")
    args = parser.parse_args()

    data_file = args.data_path
    save_file = args.save_file
    mode = args.mode
    Rcom = args.rcom
    Rcapt = args.rcapt
    path_cplex = args.cplex

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
    
        # Maxime

    elif mode == "tabu":

        instance = Instance.from_disk(data_file, Rcapt=Rcapt, Rcom=Rcom)
        tabu = TabuSearch(instance)
        tabu.GenerateInitialSolution(instance)
        tabu.display(instance)
        print("resultat tabu : ", tabu.value())

    elif mode == "test":
        ()
        # Partie ou on mets des trucs temporaires pour les tester



#    print(milp.captors)
#instance.display()

# grid_size = (4, 4)
# data_folder = "data/"
# data_file = data_folder + "grille44_toy.dat"
#
# instance = Instance.from_disk(data_file, Rcom=2)
# instance.display()

start = time.time()
# solution = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3)])
# solution1 = Solution([(1, 0), (2, 0), (3, 0), (3, 2), (3, 3), (0, 1), (0, 2), (1, 2), (1, 3), (6, 6), (6, 7)])
# solution1 = TrivialSolution(instance)
# # solution1.find_connected_components(instance)
# print(solution1.is_valid(instance))
# print("Trivial Solution value : ", solution1.value())
# solution1.display(instance)

# solution3 = TrivialSolutionRandomized(instance)
# print(solution3.is_valid(instance))
# print("Trivial Randomized Solution value : ", solution3.value())
# solution3.display(instance)
