import sys
from collections import deque, defaultdict

def bfs(start, end, graph):
    queue = deque([(start, 0)])
    visited = set([start])
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

def max_distance(N1, N2, M, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    # Find the shortest path from 1 to N1 and from N1+1 to N1+N2
    dist1 = bfs(1, N1, graph)
    dist2 = bfs(N1 + 1, N1 + N2, graph)

    max_dist = dist1 + dist2 + 1

    # Check all possible edges between 1 to N1 and N1+1 to N1+N2
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            if v not in graph[u]:
                # Calculate the new distance if we add the edge (u, v)
                new_dist1 = bfs(1, u, graph)
                new_dist2 = bfs(v, N1 + N2, graph)
                new_dist = new_dist1 + new_dist2 + 1
                max_dist = max(max_dist, new_dist)

    return max_dist

# Read input
input = sys.stdin.read
data = input().split()

N1 = int(data[0])
N2 = int(data[1])
M = int(data[2])

edges = []
for i in range(3, 3 + 2 * M, 2):
    a = int(data[i])
    b = int(data[i + 1])
    edges.append((a, b))

# Calculate and print the result
result = max_distance(N1, N2, M, edges)
print(result)