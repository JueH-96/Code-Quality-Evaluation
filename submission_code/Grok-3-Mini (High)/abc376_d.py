import sys
import collections

def bfs(graph, start):
    dist = [-1] * len(graph)
    dist[start] = 0
    q = collections.deque()
    q.append(start)
    while q:
        u = q.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2

# Create graphs
graph = [[] for _ in range(N + 1)]
graph_rev = [[] for _ in range(N + 1)]
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2
    graph[a].append(b)
    graph_rev[b].append(a)  # Add reverse edge for graph_rev

# Compute distances
dist_out = bfs(graph, 1)  # Distance from 1 to all nodes
dist_in = bfs(graph_rev, 1)  # Distance from all nodes to 1 in original graph

# Find minimum cycle length containing vertex 1
min_cycle = float('inf')
for u in range(1, N + 1):
    if u != 1:
        if dist_out[u] != -1 and dist_in[u] != -1:
            sum_dist = dist_out[u] + dist_in[u]
            if sum_dist < min_cycle:
                min_cycle = sum_dist

# Output the result
if min_cycle == float('inf'):
    print(-1)
else:
    print(min_cycle)