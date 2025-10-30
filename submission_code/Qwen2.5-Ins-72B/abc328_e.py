import sys
from itertools import combinations

def find_min_spanning_tree_cost(N, M, K, edges):
    def kruskal(edges, N, K):
        parent = list(range(N))
        rank = [0] * N

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                if rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1

        edges.sort(key=lambda x: x[2])
        cost = 0
        for u, v, w in edges:
            if find(u) != find(v):
                union(u, v)
                cost = (cost + w) % K
        return cost

    return kruskal(edges, N, K)

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])
edges = [(int(data[3 * i + 3]) - 1, int(data[3 * i + 4]) - 1, int(data[3 * i + 5])) for i in range(M)]

print(find_min_spanning_tree_cost(N, M, K, edges))