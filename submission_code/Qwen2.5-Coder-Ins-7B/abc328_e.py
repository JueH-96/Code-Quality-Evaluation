import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
for i in range(M):
    u = int(data[3 + 3 * i]) - 1
    v = int(data[4 + 3 * i]) - 1
    w = int(data[5 + 3 * i])
    edges.append((w, u, v))

# Kruskal's algorithm to find the minimum spanning tree
parent = list(range(N))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rootX != rootY:
        parent[rootX] = rootY

edges.sort()
min_cost = 0

for w, u, v in edges:
    if find(u) != find(v):
        union(u, v)
        min_cost = (min_cost + w) % K

print(min_cost)