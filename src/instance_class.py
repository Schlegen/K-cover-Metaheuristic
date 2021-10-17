from utils.build_grid import extract_points, generate_grid, delete_captors
import matplotlib.pyplot as plt
import numpy as np

class Instance:

    def __init__(self, grid, n_deleted_points, Rcapt=1, Rcom=1):
        """constructeur de la classe Instance

        Args:
            grid (objet de dimension_grille): tableau comprennant des 1 pour les sommets à capter, un 2 pour le dépot et un 0 pour les sommets n'appartenant pas à la grille 
            Rcapt (float, optionnal): rayon de captage. Defaults to None.
            Rcom (float, optionnal): rayon de communication. Defaults to None.
        """
        self.grid = grid
        self.n_deleted_points = n_deleted_points
        self.x, self.y = grid.shape
        self.Rcapt = Rcapt
        self.Rcom = Rcom
        
    @classmethod
    def from_disk(cls, data_file, size=(10,10), Rcapt=1, Rcom=1):
        captors_to_delete = extract_points(data_file)
        grid = generate_grid(size)
        grid = delete_captors(grid, captors_to_delete)
        return cls(grid, len(captors_to_delete), Rcapt, Rcom)

    def display(self):
        for i in range(self.x):
            for j in range(self.y):
                if self.grid[i, j] == 1:
                    plt.scatter(i, j, marker="+", color='blue')
                elif self.grid[i, j] == 2:
                    plt.scatter(i, j, marker="o", color='red')
        plt.show()

