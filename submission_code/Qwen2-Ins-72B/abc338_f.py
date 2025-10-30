from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import bellman_ford

N, M = map(int, input().split())
U, V, W = [], [], []
for _ in range(M):
    u, v, w = map(int, input().split())
    U.append(u-1)
    V.append(v-1)
    W.append(w)

graph = csr_matrix((W, (U, V)), shape=(N, N))
_, has_path = bellman_ford(graph, indices=0, return_predecessors=True, unweighted=True)

if not all(has_path):
    print("No")
else:
    print(graph[has_path, :][:, has_path].min())