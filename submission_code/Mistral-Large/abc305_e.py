import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
K = int(data[index + 2])
index += 3

edges = []
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2
    edges.append((a, b))

guards = []
for _ in range(K):
    p = int(data[index])
    h = int(data[index + 1])
    index += 2
    guards.append((p, h))

# Build the graph
graph = defaultdict(list)
for a, b in edges:
    graph[a].append(b)
    graph[b].append(a)

# BFS to find all guarded vertices
guarded_vertices = set()

for p, h in guards:
    queue = deque([(p, 0)])
    visited = set([p])
    while queue:
        current, dist = queue.popleft()
        if dist > h:
            break
        guarded_vertices.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

# Output the result
guarded_vertices = sorted(guarded_vertices)
print(len(guarded_vertices))
print(" ".join(map(str, guarded_vertices)))