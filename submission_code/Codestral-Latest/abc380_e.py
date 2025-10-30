import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

# Initialize the color of each cell
color = list(range(1, N + 1))

# Initialize the union-find data structure
parent = list(range(N + 1))
size = [1] * (N + 1)

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootX = find(x)
    rootY = find(y)
    if rootX != rootY:
        if size[rootX] < size[rootY]:
            rootX, rootY = rootY, rootX
        parent[rootY] = rootX
        size[rootX] += size[rootY]

# Process the queries
results = []
for _ in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        x = int(data[index + 1])
        c = int(data[index + 2])
        index += 3
        if color[x - 1] == c:
            if x > 1 and color[x - 2] == c:
                union(x - 1, x)
            if x < N and color[x] == c:
                union(x, x + 1)
    elif query_type == 2:
        c = int(data[index + 1])
        index += 2
        count = sum(1 for i in range(1, N + 1) if color[find(i)] == c)
        results.append(count)

# Output the results
for result in results:
    print(result)