# YOUR CODE HERE
from collections import deque

def bfs(graph, start, stamina):
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    guarded = set([start])

    while queue:
        vertex, distance = queue.popleft()
        if distance < stamina:
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    guarded.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return guarded

# Read input
N, M, K = map(int, input().split())

# Create graph
graph = {i: set() for i in range(1, N + 1)}
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].add(b)
    graph[b].add(a)

# Process guards
guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# Find guarded vertices
guarded_vertices = set()
for p, h in guards:
    guarded_vertices.update(bfs(graph, p, h))

# Sort and output result
guarded_vertices = sorted(guarded_vertices)
print(len(guarded_vertices))
print(*guarded_vertices)