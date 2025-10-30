from collections import defaultdict, deque

n, m = map(int, input().split())

# Build adjacency list with both forward and reverse edges
graph = defaultdict(list)
for _ in range(m):
    a, b, x, y = map(int, input().split())
    graph[a].append((b, x, y))
    graph[b].append((a, -x, -y))

# BFS to find positions
positions = [None] * (n + 1)
positions[1] = (0, 0)
queue = deque([1])

while queue:
    current = queue.popleft()
    cx, cy = positions[current]
    
    for neighbor, dx, dy in graph[current]:
        if positions[neighbor] is None:
            positions[neighbor] = (cx + dx, cy + dy)
            queue.append(neighbor)

# Output
for i in range(1, n + 1):
    if positions[i] is None:
        print("undecidable")
    else:
        print(positions[i][0], positions[i][1])