import numpy as np


def dist(u, v):
    """Distance entre deux points de R2

    Args:
        u (tuple): premier vecteur
        v (tuple): second vecteur
    """

    return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def dist_point_to_list(u, L):
    """
        For each point v of L, compute dist(u, v).
        Return the sorted listed of thoses distances.

    Args :
        u (tuple): point
        L (list of tuples) : list of points
    """
    array_u = np.array([list(u)] * len(L))
    array_points = np.array(L)

    array_dist = np.sqrt(
        (array_points[:, 0] - array_u[:, 0]) ** 2 +
        (array_points[:, 1] - array_u[:, 1]) ** 2
    )

    dist_and_coordo = [[array_dist[i], L[i]] for i in range(len(L))]
    return sorted(dist_and_coordo)
