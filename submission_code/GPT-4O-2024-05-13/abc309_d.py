# YOUR CODE HERE
import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N1 = int(data[0])
N2 = int(data[1])
M = int(data[2])

edges = []
for i in range(M):
    a = int(data[3 + 2 * i])
    b = int(data[4 + 2 * i])
    edges.append((a, b))

# Create adjacency list
graph = defaultdict(list)
for a, b in edges:
    graph[a].append(b)
    graph[b].append(a)

# Function to perform BFS and find shortest path lengths from a given start node
def bfs(start, n):
    dist = [-1] * (n + 1)
    queue = deque([start])
    dist[start] = 0
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist

# Find shortest path lengths from vertex 1 and vertex N1+N2
dist_from_1 = bfs(1, N1 + N2)
dist_from_N1N2 = bfs(N1 + N2, N1 + N2)

# Find the maximum possible d
max_d = 0
for u in range(1, N1 + 1):
    for v in range(N1 + 1, N1 + N2 + 1):
        if dist_from_1[u] != -1 and dist_from_N1N2[v] != -1:
            max_d = max(max_d, dist_from_1[u] + 1 + dist_from_N1N2[v])

print(max_d)