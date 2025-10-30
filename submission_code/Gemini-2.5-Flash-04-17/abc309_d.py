import sys
import collections

# Function for BFS
def bfs(start_node, N, adj):
    # dist[i] will store the shortest distance from start_node to i, or -1 if unreachable
    dist = [-1] * (N + 1)
    q = collections.deque()

    dist[start_node] = 0
    q.append(start_node)

    while q:
        u = q.popleft()

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

# Read input
# Use sys.stdin.readline for faster input
input = sys.stdin.readline
N1, N2, M = map(int, input().split())
N = N1 + N2

# Build adjacency list
# Using 1-based indexing for vertices
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

# BFS from vertex 1 to find distances to all reachable nodes in its component
# Based on the problem statement guarantees and the fact that 1 and N are disconnected,
# vertex 1 is in a component (let's call it C1) that contains all vertices {1, ..., N1}.
# There are no edges between {1, ..., N1} and {N1+1, ..., N} in the original graph.
# The BFS from 1 will thus compute shortest distances within C1 (and potentially other vertices
# not in V1 but in C1, if any, though problem constraints imply V1 and V2 partition the vertices).
# Specifically, it computes distances from 1 to all vertices in {1, ..., N1}.
dist1 = bfs(1, N, adj)

# Find max distance from 1 to any vertex in V1 = {1, ..., N1}
# All vertices in V1 are guaranteed to be connected to 1 within the original graph structure,
# so dist1[i] should be >= 0 for i in [1, N1].
max_d1 = 0
for i in range(1, N1 + 1):
    max_d1 = max(max_d1, dist1[i])

# BFS from vertex N = N1 + N2 to find distances to all reachable nodes in its component
# Similarly, vertex N is in a component (let's call it C2) that contains all vertices {N1+1, ..., N}.
# This BFS computes distances from N to all vertices in {N1+1, ..., N}.
distN = bfs(N, N, adj)

# Find max distance from N to any vertex in V2 = {N1+1, ..., N}
# All vertices in V2 are guaranteed to be connected to N within the original graph structure,
# so distN[i] should be >= 0 for i in [N1+1, N].
max_dN = 0
for i in range(N1 + 1, N + 1):
    max_dN = max(max_dN, distN[i])

# After adding an edge (u, v) where u in V1 and v in V2, the shortest path from 1 to N
# must use this new edge. The path will be 1 --(shortest path in C1)--> u --(new edge)--> v
# --(shortest path in C2)--> N. The length of this path is dist1[u] + 1 + distN[v].
# To maximize the shortest path length d in the resulting graph, we need to maximize
# dist1[u] + 1 + distN[v] over all possible choices of u in V1 and v in V2.
# This is achieved by choosing u that maximizes dist1[u] and v that maximizes distN[v]
# independently. The maximum possible d is max(dist1[u] + 1 + distN[v] for u in V1, v in V2)
# which equals max(dist1[u] for u in V1) + 1 + max(distN[v] for v in V2).
print(max_d1 + max_dN + 1)