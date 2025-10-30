# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

edges = []
index = 2
for i in range(1, Q + 1):
    L = int(data[index])
    R = int(data[index + 1])
    C = int(data[index + 2])
    for j in range(L, R + 1):
        edges.append((C, N + i, j))
    index += 3

# Kruskal's algorithm to find MST
edges.sort()
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

mst_cost = 0
edges_used = 0

for cost, u, v in edges:
    if find(u) != find(v):
        union(u, v)
        mst_cost += cost
        edges_used += 1

# Check if all nodes are connected
root = find(1)
connected = all(find(i) == root for i in range(1, N + Q + 1))

if connected:
    print(mst_cost)
else:
    print(-1)