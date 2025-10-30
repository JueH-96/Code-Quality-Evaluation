# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import defaultdict, deque

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

data = input().split()
index = 0

N = int(data[index])
index += 1
M = int(data[index])
index += 1

edges = []
for _ in range(M):
    u = int(data[index])
    index += 1
    v = int(data[index])
    index += 1
    edges.append((u, v))

K = int(data[index])
index += 1

forbidden_pairs = []
for _ in range(K):
    x = int(data[index])
    index += 1
    y = int(data[index])
    index += 1
    forbidden_pairs.append((x, y))

Q = int(data[index])
index += 1

queries = []
for _ in range(Q):
    p = int(data[index])
    index += 1
    q = int(data[index])
    index += 1
    queries.append((p, q))

# Initialize Union-Find structure
parent = list(range(N + 1))
rank = [0] * (N + 1)

# Add initial edges to the Union-Find structure
for u, v in edges:
    union(parent, rank, u, v)

# Check each query
results = []
for p, q in queries:
    # Temporarily add the edge (p, q)
    temp_parent = parent[:]
    temp_rank = rank[:]
    union(temp_parent, temp_rank, p, q)
    
    is_good = True
    for x, y in forbidden_pairs:
        if find(temp_parent, x) == find(temp_parent, y):
            is_good = False
            break
    
    if is_good:
        results.append("Yes")
    else:
        results.append("No")

print("
".join(results))