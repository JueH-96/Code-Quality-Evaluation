import sys
from collections import deque

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Build the graph with both directions and offsets
graph = [[] for _ in range(N + 1)]
for _ in range(M):
    A = int(data[index])
    index += 1
    B = int(data[index])
    index += 1
    X = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1
    # Add directed edge from A to B with offset (X, Y)
    graph[A].append((B, X, Y))
    # Add directed edge from B to A with offset (-X, -Y)
    graph[B].append((A, -X, -Y))

# BFS to compute positions starting from person 1
visited = [False] * (N + 1)
pos_x = [None] * (N + 1)
pos_y = [None] * (N + 1)

# Initialize person 1 at (0, 0)
pos_x[1] = 0
pos_y[1] = 0
visited[1] = True
queue = deque([1])

while queue:
    u = queue.popleft()
    for neighbor in graph[u]:
        v, dx, dy = neighbor
        if not visited[v]:
            # Compute position of v based on u's position and offset
            pos_x[v] = pos_x[u] + dx
            pos_y[v] = pos_y[u] + dy
            visited[v] = True
            queue.append(v)

# Output the results for each person
for i in range(1, N + 1):
    if visited[i]:
        print(f"{pos_x[i]} {pos_y[i]}")
    else:
        print("undecidable")