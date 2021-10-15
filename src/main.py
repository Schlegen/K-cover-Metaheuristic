from utils.build_grid import extract_and_build
from utils.display_grid import display

grid_size = (10, 10)
data_folder = "data/"
data_file = data_folder + "grille1010_2.dat"

grid = extract_and_build(data_file, grid_size)
display(grid)
