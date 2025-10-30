import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
K = int(data[index + 2])
index += 3

# Build the graph
graph = defaultdict(list)
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    graph[a].append(b)
    graph[b].append(a)
    index += 2

# Guards information
guards = []
for _ in range(K):
    p = int(data[index])
    h = int(data[index + 1])
    guards.append((p, h))
    index += 2

# BFS to find all vertices within h steps from each guard
def bfs(start, h):
    queue = deque([(start, 0)])
    visited = set()
    guarded_vertices = set()

    while queue:
        vertex, distance = queue.popleft()
        if distance > h:
            continue
        if vertex not in visited:
            visited.add(vertex)
            guarded_vertices.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    return guarded_vertices

# Find all guarded vertices
all_guarded = set()
for p, h in guards:
    all_guarded.update(bfs(p, h))

# Output the result
guarded_list = sorted(all_guarded)
print(len(guarded_list))
print(" ".join(map(str, guarded_list)))