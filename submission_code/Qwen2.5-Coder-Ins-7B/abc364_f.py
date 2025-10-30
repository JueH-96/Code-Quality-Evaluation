import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

# Union-Find structure
parent = list(range(N + Q + 1))
rank = [0] * (N + Q + 1)

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

# List to store edges
edges = []

# Read and process each query
index = 2
for i in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    C = int(data[index + 2])
    for j in range(L, R + 1):
        edges.append((C, N + i, j))
    index += 3

# Sort edges by cost
edges.sort()

# Kruskal's algorithm to find the minimum spanning tree
mst_cost = 0
num_edges = 0
for cost, u, v in edges:
    if find(u) != find(v):
        union(u, v)
        mst_cost += cost
        num_edges += 1
        if num_edges == N + Q - 1:
            break

# Check if the graph is connected
if find(1) != find(N + Q):
    print(-1)
else:
    print(mst_cost)