import sys
from collections import defaultdict, deque

# Read input
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
books = []
index = 1
for i in range(N):
    C = int(data[index])
    P = list(map(int, data[index + 1:index + 1 + C]))
    books.append((C, P))
    index += 1 + C

# Build graph and in-degree array
graph = defaultdict(list)
in_degree = [0] * (N + 1)
for i in range(1, N + 1):
    C, P = books[i - 1]
    for j in P:
        graph[j].append(i)
        in_degree[i] += 1

# Topological sort
queue = deque([i for i in range(1, N + 1) if in_degree[i] == 0])
order = []

while queue:
    node = queue.popleft()
    if node == 1:
        continue
    order.append(node)
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

# Print result
print(" ".join(map(str, order)))