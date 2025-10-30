import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
K = int(data[index + 2])
index += 3

graph = defaultdict(list)

# Read edges
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2
    graph[a].append(b)
    graph[b].append(a)

guards = []
# Read guards
for _ in range(K):
    p = int(data[index])
    h = int(data[index + 1])
    index += 2
    guards.append((p, h))

guarded_vertices = set()

# BFS function to find all vertices within distance h from start vertex p
def bfs(start, max_distance):
    queue = deque([(start, 0)])  # (current vertex, current distance)
    visited = set([start])
    while queue:
        current, dist = queue.popleft()
        if dist <= max_distance:
            guarded_vertices.add(current)
            if dist < max_distance:
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

# Perform BFS for each guard
for p, h in guards:
    bfs(p, h)

# Prepare output
guarded_list = sorted(guarded_vertices)
print(len(guarded_list))
if guarded_list:
    print(" ".join(map(str, guarded_list)))