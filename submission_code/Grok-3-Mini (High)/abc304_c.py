import sys
import collections

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
D = int(data[index + 1])
index += 2

# Read positions
positions = []
for _ in range(N):
    x = int(data[index])
    y = int(data[index + 1])
    positions.append((x, y))
    index += 2

# Build adjacency list
adj = [[] for _ in range(N)]
D_sq = D * D
for i in range(N):
    for j in range(i + 1, N):
        dx = positions[i][0] - positions[j][0]
        dy = positions[i][1] - positions[j][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq <= D_sq:
            adj[i].append(j)
            adj[j].append(i)

# BFS to find all infected people starting from person 1 (index 0)
visited = [False] * N
queue = collections.deque()
queue.append(0)
visited[0] = True

while queue:
    current = queue.popleft()
    for neighbor in adj[current]:
        if not visited[neighbor]:
            visited[neighbor] = True
            queue.append(neighbor)

# Output the result for each person
for i in range(N):
    if visited[i]:
        print("Yes")
    else:
        print("No")