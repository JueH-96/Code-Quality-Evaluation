import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

n = int(data[0])
edges = [tuple(map(int, data[i].split())) for i in range(1, n)]

# Build the graph
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# Calculate the degree of each node
degree = [0] * (n + 1)
for u, v in edges:
    degree[u] += 1
    degree[v] += 1

# Initialize the queue with leaves (nodes with degree 1)
queue = deque([i for i in range(1, n + 1) if degree[i] == 1])

# Perform the operations
operations = 0
while queue:
    leaf = queue.popleft()
    if leaf == 1:
        break
    operations += 1
    for neighbor in graph[leaf]:
        degree[neighbor] -= 1
        if degree[neighbor] == 1:
            queue.append(neighbor)

print(operations)