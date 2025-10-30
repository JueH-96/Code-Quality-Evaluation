# YOUR CODE HERE
def find(parent, i):
    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    
    if rootX != rootY:
        if rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        elif rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def kruskal_minimum_spanning_tree(n, edges, k):
    # Sort edges based on weight
    edges.sort(key=lambda x: x[2])
    
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    mst_weight = 0
    edge_count = 0
    
    for u, v, weight in edges:
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst_weight += weight
            edge_count += 1
            if edge_count == n - 1:
                break
    
    return mst_weight % k

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
index = 3
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    w = int(data[index + 2])
    edges.append((u, v, w))
    index += 3

result = kruskal_minimum_spanning_tree(N, edges, K)
print(result)