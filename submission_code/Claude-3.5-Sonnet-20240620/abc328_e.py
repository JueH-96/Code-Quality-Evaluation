# YOUR CODE HERE
import sys
from itertools import combinations

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def kruskal(N, edges, K):
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    min_cost = float('inf')

    for tree_edges in combinations(edges, N - 1):
        cost = 0
        parent = list(range(N + 1))
        rank = [0] * (N + 1)
        valid_tree = True

        for u, v, w in tree_edges:
            x = find(parent, u)
            y = find(parent, v)
            if x != y:
                union(parent, rank, x, y)
                cost = (cost + w) % K
            else:
                valid_tree = False
                break

        if valid_tree:
            min_cost = min(min_cost, cost)

    return min_cost

N, M, K = map(int, input().split())
edges = []
for _ in range(M):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

result = kruskal(N, edges, K)
print(result)