from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np
import sys
input = sys.stdin.readline

N, D = map(int, input().split())
D = D ** 2
XY = [list(map(int, input().split())) for _ in range(N)]

graph = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        x1, y1 = XY[i]
        x2, y2 = XY[j]
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= D:
            graph[i][j] = 1
            graph[j][i] = 1

graph = csr_matrix(graph)
dist = shortest_path(graph, directed=False, unweighted=True, indices=0)

for i in range(N):
    if dist[i] == np.inf:
        print('No')
    else:
        print('Yes')