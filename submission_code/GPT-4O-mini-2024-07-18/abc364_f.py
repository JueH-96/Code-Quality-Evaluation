import sys
from collections import defaultdict
import heapq

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def kruskal(n, edges):
    parent = [i for i in range(n + 1)]
    rank = [0] * (n + 1)
    edges.sort(key=lambda x: x[2])  # Sort edges by cost
    total_cost = 0
    edges_used = 0

    for u, v, cost in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            total_cost += cost
            edges_used += 1

    # Check if all vertices are connected
    root = find(parent, 1)
    for i in range(1, n + 1):
        if find(parent, i) != root:
            return -1  # Not all vertices are connected

    return total_cost

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N, Q = map(int, data[0].split())
    edges = []
    
    for i in range(1, Q + 1):
        L, R, C = map(int, data[i].split())
        for j in range(L, R + 1):
            edges.append((N + i, j, C))

    # Calculate the minimum spanning tree cost
    result = kruskal(N + Q, edges)
    print(result)

if __name__ == "__main__":
    main()