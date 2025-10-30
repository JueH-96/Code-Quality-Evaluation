import sys
input = sys.stdin.read

def find(parent, i):
    if parent[i] == i:
        return i
    else:
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

def kruskal(edges, n):
    edges.sort(key=lambda x: x[2])
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    mst_cost = 0
    for u, v, cost in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_cost += cost
    root = find(parent, 1)
    for i in range(1, n + 1):
        if find(parent, i) != root:
            return -1
    return mst_cost

data = input().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

edges = []
for i in range(Q):
    L = int(data[index])
    index += 1
    R = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    for j in range(L, R + 1):
        edges.append((j, N + i, C))

result = kruskal(edges, N + Q)
print(result)