import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools


num_node = 55
self_link = [(i, i) for i in range(num_node)]
inward = [(4, 3), (3, 2), (2, 1), (7, 6), (6, 5), (5, 1), (8, 1), (1, 0), (9, 0), (10, 0), (11, 9), (12, 0),
          (13, 14), (14, 15), (15, 16), (16, 17), (13, 18), (18, 19), (19, 20), (20, 21), (13, 22), (22, 23), (23, 24), (24, 25), (13, 26), (26, 27), (27, 28), (28, 29), (13, 30), (30, 31), (31, 32), (32, 33),
          (34, 35), (35, 36), (36, 37), (37, 38), (34, 39), (39, 40), (40, 41), (41, 42), (34, 43), (43, 44), (44, 45), (45, 46), (34, 47), (47, 48), (48, 49), (49, 50), (34, 51), (51, 52), (52, 53), (53, 54),
          ]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
