import sys
import collections

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read N and M
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Total nodes in bipartite graph: N set nodes and M element nodes
V = N + M

# Adjacency list for bipartite graph
adj = [[] for _ in range(V)]

# Flags to check if a set contains element 1 or element M
has_one = [False] * N
has_M = [False] * N

# Read all sets and build the bipartite graph
for i in range(N):
    A = int(data[index])
    index += 1
    for _ in range(A):
        e = int(data[index])
        index += 1
        # Element node index for element e (1-based)
        elem_node = N + e - 1
        # Add edges in both directions
        adj[i].append(elem_node)
        adj[elem_node].append(i)
        # Set flags for set i
        if e == 1:
            has_one[i] = True
        if e == M:
            has_M[i] = True

# Check if there is no set containing 1 or no set containing M
if not any(has_one) or not any(has_M):
    print(-1)
    sys.exit()

# BFS to find shortest path in bipartite graph
dist = [-1] * V
queue = collections.deque()

# Enqueue all set nodes that contain element 1 with distance 0
for i in range(N):
    if has_one[i]:
        dist[i] = 0
        queue.append(i)

found = False
min_dist_bip = -1

while queue:
    u = queue.popleft()
    # Check if u is a set node and contains element M
    if u < N and has_M[u]:
        found = True
        min_dist_bip = dist[u]
        break  # No need to continue, found the minimum distance
    # Enqueue unvisited neighbors
    for v in adj[u]:
        if dist[v] == -1:  # Not visited
            dist[v] = dist[u] + 1
            queue.append(v)

# After BFS, check if a path was found
if found:
    min_merges = min_dist_bip // 2
    print(min_merges)
else:
    print(-1)