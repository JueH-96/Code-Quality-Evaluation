import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

edges = []

for _ in range(M):
    K = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    vertices = [int(data[index + i]) for i in range(K)]
    index += K

    for i in range(K):
        for j in range(i + 1, K):
            edges.append((C, vertices[i], vertices[j]))

edges.sort()

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

mst_weight = 0
components = N

for weight, u, v in edges:
    if find(u) != find(v):
        union(u, v)
        mst_weight += weight
        components -= 1

if components == 1:
    print(mst_weight)
else:
    print(-1)