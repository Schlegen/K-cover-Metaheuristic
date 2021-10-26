#!/bin/bash


path_to_cplex="/opt/ibm/ILOG/CPLEX_Studio201/cplex/bin/x86-64_linux/cplex"
data_list=("data/grille1010_1.dat" "data/grille1010_2.dat" "data/grille1515_1.dat" "data/grille1515_2.dat" "data/grille2020_1.dat" "data/grille2020_2.dat" "data/grille2525_1.dat" "data/grille2525_2.dat" "data/grille3030_1.dat" "data/grille3030_2.dat" "data/grille33_toy.dat" "data/grille4040_1.dat" "data/grille4040_2.dat" "data/grille44_toy.dat" "data/captANOR150_7_4.dat")

# for file in ${data_list[@]};
# do
#     for R in "1 1" "1 2" "2 2" "2 3"; do 
#         set -- $R
#         tmp=${file#*/}   # remove prefix ending in "/"
#         log_name=${tmp%.*}   # remove suffix starting with "."
#         python3 main.py -m "bound" -d $file -rcapt $1 -rcom $2 -s "saves/scores.csv" --cplex $path_to_cplex> "logs/logs_bound/${log_name}_$1_$2.txt"
#     done 
# done

for file in "data/grille44_toy.dat" "data/captANOR150_7_4.dat";
do
    for R in "1 1" "1 2" "2 2" "2 3"; do 
        set -- $R
        tmp=${file#*/}   # remove prefix ending in "/"
        log_name=${tmp%.*}   # remove suffix starting with "."
        python3 main.py -m "bound" -d $file -rcapt $1 -rcom $2 -s "saves/scores.csv" --cplex $path_to_cplex> "logs/logs_bound/${log_name}_$1_$2.txt"
    done 
done