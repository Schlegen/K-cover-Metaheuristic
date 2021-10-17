from copy import deepcopy
from instance_class import Instance
from utils.errors import Error, InputError
import math as mh
import matplotlib.pyplot as plt

class Solution:

    def __init__(self, captors):
        """constructeur de la classe Solution

        Args:
            captors (list of tuples): liste des coordonnées des capteurs dans la solution
        """        
        self.captors = captors

    def is_valid(self, instance):
        """verifie que la solution respecte bien les contraintes de l'instance

        Args:
            instance (Instance): instance à laquelle correspond la solution
        """
        void_grid = deepcopy(instance.grid)


        Rmax = int(max(instance.Rcapt, instance.Rcom))

        #on dispose les capteurs dans la grille :
        for captor in self.captors:
            if captor[0] < 0 or captor[1] < 0 or captor[0] > instance.x or captor[1] > instance.y or void_grid[captor[0], captor[1]] == 0:
                raise InputError(f"Solution invalide : On ne peut pas placer le capteur ({captor[0]},{captor[1]})")
            else:
                void_grid[captor[0],captor[1]] = 3

        captors_to_treat = [] # liste des sommets qui communiquent avec le puit
        n_covered_points = 0 # nombre de points couverts par les capteurs

        # initialisation
        for i in range (0, int(instance.Rcom)+1):
            for j in range (0, int(instance.Rcom)+1):
                if i < instance.x and j < instance.y and mh.sqrt(i ** 2 + j **2) <= instance.Rcom and void_grid[i,j] == 3:
                    captors_to_treat.append([i, j])
                    n_covered_points += 1
                    void_grid[i, j] = 0

        #Pour chaque capteur de la liste

        while len(captors_to_treat) > 0:
            captor = captors_to_treat.pop()
            for i in range (-Rmax, Rmax + 1):
                for j in range (-Rmax, Rmax + 1):
                    dist_to_captor = mh.sqrt(i ** 2 + j **2)
                    if captor[0] + i < instance.x and captor[1] + j < instance.y and captor[0] + i >= 0 and captor[1] + j >= 0:
                        if dist_to_captor <= instance.Rcapt and (void_grid[captor[0] + i, captor[1] + j] == 1 or void_grid[captor[0] + i, captor[1] + j] == 2):
                            n_covered_points += 1
                            void_grid[captor[0] + i, captor[1] + j] = 0
                        elif dist_to_captor <= instance.Rcom and void_grid[captor[0] + i, captor[1] + j] == 3:
                            captors_to_treat.append((captor[0] + i, captor[1] + j))
                            n_covered_points += 1
                            void_grid[captor[0] + i, captor[1] + j] = 0
        return n_covered_points == instance.x * instance.y - instance.n_deleted_points

    def value(self):
        return len(self.captors)

    def display(self, instance):

        ## A enrichir pour afficher les liens de communication et de captation
        for i in range(instance.x):
            for j in range(instance.y):
                if instance.grid[i, j] == 1:
                    plt.scatter(i, j, marker="+", color='blue')
                elif instance.grid[i, j] == 2:
                    plt.scatter(i, j, marker="o", color='red')

        for captor in self.captors : 
            plt.scatter(captor[0], captor[1], marker="D", color='orange')
        plt.show()
