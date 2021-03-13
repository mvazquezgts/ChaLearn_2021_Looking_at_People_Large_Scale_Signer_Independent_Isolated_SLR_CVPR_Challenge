import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 55
self_link = [(i, i) for i in range(num_node)]

inward = [(0, 1),(0, 4),(0, 7),(0, 8),(1, 8),(1, 4),(1, 7),(7, 4),(7, 8),(4, 8),(2, 1),(3, 2),(4, 5),(5, 6),(9, 8),(9, 10),(10, 8),(11, 9),(12, 10),
          (13, 14),(13, 18),(13, 22),(13, 26),(13, 30),(14, 18),(14, 22),(14, 26),(14, 30),(18, 22),(18, 26),(18, 30),(22, 26),(22, 30),(26, 30),(14, 15),(15, 16),(16, 17),(18, 19),(19, 20),(20, 21),(22, 23),(23, 24),(24, 25),(26, 27),(27, 28),(28, 29),(30, 31),(31, 32),(32, 33),
          (34, 35),(34, 39),(34, 43),(34, 47),(34, 51),(35, 39),(35, 43),(35, 47),(35, 51),(39, 43),(39, 47),(39, 51),(43, 47),(43, 51),(47, 51),(35, 36),(36, 37),(37, 38),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(47, 48),(48, 49),(49, 50),(51, 52),(52, 53),(53, 54)
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
