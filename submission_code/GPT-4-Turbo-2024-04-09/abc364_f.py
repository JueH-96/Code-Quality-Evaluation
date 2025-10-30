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
    index += 3
    for j in range(L, R + 1):
        # Edge from vertex N+i (which is N+i) to vertex j with cost C
        edges.append((C, N + i, j))

# Sort edges by cost for Kruskal's algorithm
edges.sort()

# Union-Find data structure
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
        return True
    return False

# Kruskal's algorithm to find MST
mst_cost = 0
edges_used = 0
for cost, u, v in edges:
    if union(u, v):
        mst_cost += cost
        edges_used += 1
        # If we have used N+Q-1 edges, the MST is complete
        if edges_used == N + Q - 1:
            print(mst_cost)
            break
else:
    # If we exit the loop without having used enough edges, the graph is disconnected
    print(-1)