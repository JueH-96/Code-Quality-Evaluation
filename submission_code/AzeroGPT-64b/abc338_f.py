import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

N, M = map(int, input().split())
edges = np.zeros((M, 3), dtype=np.intc)
for i in range(M):
  u, v, w = map(int, input().split())
  edges[i] = [u-1, v-1, w]

src = np.concatenate((edges[:,0], edges[:,1]))
dst = np.concatenate((edges[:,1], edges[:,0]))
weights = np.concatenate((edges[:,2], edges[:,2]))
graph = coo_matrix((weights, (src, dst)), shape=(N, N)).tocsr()

dists = dijkstra(csgraph=graph, directed=True, indices=range(N), unweighted=False)
if np.any(np.isinf(dists)):
  print("No")
else:
  min_total_weight = np.min(dists + dists[:, np.newaxis])
  print(int(min_total_weight))