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

def subgraph_is_connex(adj, subgraph):
    """vérifie que le sous graphe induit par un sous ensemble de sommets est connexe à l'aide d'un dfs
    Args:
        adj (np.array): matrice carrée à valeur dans 0 1 représentant la matrice d'adjacence
        subgraph (liste): tableau des indices des sommets conservés 
    """

    sub_adj = adj[np.ix_(subgraph, subgraph)]
    stack = [0]
    n = subgraph.size
    added_to_stack = np.zeros(subgraph.size, dtype=bool)
    added_vertices = 0
    while len(stack) > 0 and added_vertices < n:
        u = stack.pop()
        neighbours = np.argwhere(sub_adj[u].flatten()).flatten()
        n_neighbours = neighbours.size
        idx_v = 0
        while idx_v < n_neighbours and added_vertices < n:
            v  = neighbours[idx_v]
#            print("3", v, added_to_stack[v])
            if not added_to_stack[v]:
                added_to_stack[v] = True
                stack.append(v)
                added_vertices += 1
            idx_v += 1
    return added_vertices == n

