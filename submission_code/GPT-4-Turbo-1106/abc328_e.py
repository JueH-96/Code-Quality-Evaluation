import sys
from itertools import combinations

# Read the input
N, M, K = map(int, sys.stdin.readline().split())
edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(M)]

# Kruskal's algorithm with Union-Find
parent = list(range(N+1))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rootX != rootY:
        parent[rootY] = rootX

# Sort the edges by weight
edges.sort(key=lambda x: x[2])

# Initialize the minimum cost to a large number
min_cost = float('inf')

# Try all combinations of N-1 edges to find the minimum spanning tree
for comb in combinations(edges, N-1):
    # Reset the Union-Find structure
    parent = list(range(N+1))
    cost = 0
    for u, v, w in comb:
        if find(u) != find(v):
            union(u, v)
            cost += w
    # Check if all vertices are connected
    if len(set(find(i) for i in range(1, N+1))) == 1:
        min_cost = min(min_cost, cost % K)

# Print the minimum cost
print(min_cost)