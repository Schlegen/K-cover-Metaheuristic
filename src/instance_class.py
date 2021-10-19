from utils.build_grid import extract_points, init_grid
import matplotlib.pyplot as plt
import numpy as np


class Instance:
    def __init__(self, deleted_points, size, Rcapt=1, Rcom=1):
        """constructeur de la classe Instance

        Args:
            Rcapt (float, optionnal): rayon de captage. Defaults to 1.
            Rcom (float, optionnal): rayon de communication. Defaults to 1.
        """

        self.source = (0, 0)
        self.n = size[0]
        self.m = size[1]

        self.deleted_points = deleted_points
        self.n_deleted_points = len(deleted_points)

        self.targets = [(i, j) for i in range(self.n) for j in range(self.m) if (i, j) not in deleted_points
                        and (i, j) != self.source]
        self.n_targets = len(self.targets)

        self.grid = init_grid(deleted_points, size)

        self.Rcapt = Rcapt
        self.Rcom = Rcom
        
    @classmethod
    def from_disk(cls, data_file, size=(10, 10), Rcapt=1, Rcom=1):
        captors_to_delete = extract_points(data_file)
        return cls(captors_to_delete, size, Rcapt, Rcom)

    def draw_data(self):
        for target in self.targets:
            plt.scatter(target[0], target[1], marker="+", color='blue')
        plt.scatter(self.source[0], self.source[1], marker="o", color='red')

    def display(self):
        plt.figure("Instance")
        self.draw_data()
        plt.show()



