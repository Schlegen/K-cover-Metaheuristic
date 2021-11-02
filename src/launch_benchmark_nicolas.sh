#!/bin/bash

data_list=("data/captANOR1600_16_100_2021.dat")
#("data/grille1010_1.dat" "data/grille1010_2.dat" "data/grille1515_1.dat" "data/grille1515_2.dat" "data/grille2020_1.dat" "data/grille2020_2.dat" "data/grille2525_1.dat" "data/grille2525_2.dat" "data/grille3030_1.dat" "data/grille3030_2.dat" "data/grille33_toy.dat" "data/grille4040_1.dat" "data/grille4040_2.dat" "data/grille44_toy.dat" "data/captANOR150_7_4.dat")

for file in ${data_list[@]};
do  
    for k in 1 2 3; do
        for R in "1 1" "1 2" "2 2" "2 3"; do 
            set -- $R
            tmp=${file#*/}   # remove prefix ending in "/"
            log_name=${tmp%.*}   # remove suffix starting with "."
            echo $file $1 $2 $k
            python3 main.py -m "localsearch" -d $file -rcapt $1 -rcom $2 -k $k -s "saves/scores_benchmark_nicolas.csv" -t 600 --optim > "logs/logs_ls/${log_name}_$1_$2_$k.txt"
            python3 main.py -m "genetic" -d $file -rcapt $1 -rcom $2 -k $k -s "saves/scores_benchmark_nicolas.csv" -t 600 --optim > "logs/logs_genetic/${log_name}_$1_$2_$k.txt"
        done
    done
done