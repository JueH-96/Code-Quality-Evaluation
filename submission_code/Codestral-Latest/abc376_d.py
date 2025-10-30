import sys
from collections import defaultdict, deque

def find_min_cycle_with_vertex_1(N, M, edges):
    # Build the graph
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)

    # BFS to find the shortest path from vertex 1 to itself
    queue = deque([(1, 1, 0)])  # (current_vertex, parent_vertex, distance)
    visited = {1: 0}  # vertex: distance from vertex 1

    while queue:
        current, parent, dist = queue.popleft()

        for neighbor in graph[current]:
            if neighbor == 1:
                # Cycle detected
                return dist + 1
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, current, dist + 1))

    return -1

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = []
for i in range(2, 2 * M + 2, 2):
    a = int(data[i])
    b = int(data[i + 1])
    edges.append((a, b))

# Find and print the result
result = find_min_cycle_with_vertex_1(N, M, edges)
print(result)