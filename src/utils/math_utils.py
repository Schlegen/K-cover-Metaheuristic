import numpy as np

def dist(u, v):
    """Distance entre deux points de R2

    Args:
        u (tuple): premier vecteur
        v (tuple): second vecteur
    """

    return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)