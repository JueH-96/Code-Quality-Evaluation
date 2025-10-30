import sys
from collections import deque

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
S = int(data[index + 2])
T = int(data[index + 3])
index += 4

# Create adjacency list for the graph (1-based indexing)
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    index += 2
    adj[u].append(v)
    adj[v].append(u)  # Undirected graph, so add both directions

# BFS to find shortest path in state space (posA, posB)
queue = deque()
visited = set()
queue.append((S, T, 0))  # (position of A, position of B, distance)
visited.add((S, T))

min_dist = -1

while queue:
    cur_A, cur_B, dist = queue.popleft()
    if cur_A == T and cur_B == S:
        min_dist = dist
        break  # Found the goal state

    # Move A to adjacent vertices
    for neigh in adj[cur_A]:
        if neigh != cur_B and (neigh, cur_B) not in visited:
            queue.append((neigh, cur_B, dist + 1))
            visited.add((neigh, cur_B))  # Mark as visited when enqueued

    # Move B to adjacent vertices
    for neigh in adj[cur_B]:
        if neigh != cur_A and (cur_A, neigh) not in visited:
            queue.append((cur_A, neigh, dist + 1))
            visited.add((cur_A, neigh))  # Mark as visited when enqueued

# Output the result
if min_dist != -1:
    print(min_dist)
else:
    print(-1)