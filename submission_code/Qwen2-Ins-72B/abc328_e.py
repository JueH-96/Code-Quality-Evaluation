from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

N, M, K = map(int, input().split())
edges = np.zeros((M, 3), dtype=np.int64)
for i in range(M):
    u, v, w = map(int, input().split())
    edges[i] = [u-1, v-1, w % K]

graph = csr_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=(N, N))
graph += csr_matrix((edges[:, 2], (edges[:, 1], edges[:, 0])), shape=(N, N))

mst = minimum_spanning_tree(graph)
mst_cost = int(mst.sum()) % K

print(mst_cost)