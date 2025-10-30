# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
index = 2

for _ in range(M):
    K = int(data[index])
    C = int(data[index + 1])
    vertices = list(map(int, data[index + 2: index + 2 + K]))
    index += 2 + K
    
    for i in range(K):
        for j in range(i + 1, K):
            edges.append((C, vertices[i], vertices[j]))

# Kruskal's algorithm to find the Minimum Spanning Tree (MST)
parent = list(range(N + 1))
rank = [0] * (N + 1)

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

edges.sort()
mst_weight = 0
num_edges = 0

for weight, u, v in edges:
    if find(u) != find(v):
        union(u, v)
        mst_weight += weight
        num_edges += 1
        if num_edges == N - 1:
            break

# Check if all vertices are connected
root = find(1)
for i in range(2, N + 1):
    if find(i) != root:
        print(-1)
        sys.exit()

print(mst_weight)