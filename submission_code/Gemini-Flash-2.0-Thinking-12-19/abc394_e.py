import collections
import sys

# Read N
N = int(sys.stdin.readline())

# Read adjacency matrix C
C = [sys.stdin.readline().strip() for _ in range(N)]

# Build adjacency lists and reverse adjacency lists
# adj[u] stores list of (v, label) for edges u -> v
# rev_adj[v] stores list of (u, label) for edges u -> v
adj = [[] for _ in range(N)]
rev_adj = [[] for _ in range(N)]

for i in range(N):
    for j in range(N):
        if C[i][j] != '-':
            adj[i].append((j, C[i][j]))
            rev_adj[j].append((i, C[i][j]))

# Initialize distance matrix and queue for BFS
# dist[u][v] stores the shortest palindrome path length from u to v
# Initialize with -1 to signify unreachable
dist = [[-1] * N for _ in range(N)]
queue = collections.deque()

# Base cases: length 0 (empty path)
# A path from a vertex to itself with 0 edges has an empty label string, which is a palindrome.
# Distance is 0.
for i in range(N):
    dist[i][i] = 0
    queue.append((i, i))

# Base cases: length 1 (single edge path)
# A path of length 1 from i to j with label l has string l, which is a palindrome.
# Distance is 1.
for i in range(N):
    for j in range(N):
        # Check if there is an edge from i to j
        if C[i][j] != '-':
            # For i != j, the shortest palindrome path could be length 1.
            # dist[i][i] is already 0, so we only need to consider i != j.
            # However, the BFS structure handles minimums, so we can just add length 1 paths if
            # dist[i][j] is still -1 (meaning no shorter path found yet, which is true for i != j
            # since length 0 is only for i==j).
            if i != j and dist[i][j] == -1:
                 dist[i][j] = 1
                 queue.append((i, j))

# BFS on states (u, v) representing the pair of vertices
# The distance dist[u][v] is the length of the shortest palindrome path from u to v.
# Transitions: If we know the shortest palindrome path from u to v is length d (dist[u][v] == d),
# we can potentially form a new shortest palindrome path from s to t of length d+2
# if there is an edge s --l--> u and an edge v --l--> t for some label l.
# The path from s to t is s --l--> u ~> v --l--> t, where u ~> v is a palindrome path of length d.
# The labels of the s~>t path are l + (labels of u~>v) + l. Since labels of u~>v form a palindrome,
# this new string is also a palindrome.
while queue:
    u, v = queue.popleft()
    d = dist[u][v]

    # Iterate through all possible extensions using symmetric edges
    # An extension uses an incoming edge to u (s --l--> u) and an outgoing edge from v (v --l--> t)
    # with the same label l.
    # For each incoming edge to u (s --ls--> u):
    for s, ls in rev_adj[u]:
        # For each outgoing edge from v (v --lt--> t):
        for t, lt in adj[v]:
            # If the labels match (ls == lt), we found a valid extension
            if ls == lt:
                # The new state is (s, t) reachable with a palindrome path of length d + 2
                # If we haven't found a shorter or equal length path to (s, t) yet
                if dist[s][t] == -1:
                    dist[s][t] = d + 2
                    queue.append((s, t))

# Print the result matrix
for i in range(N):
    print(' '.join(map(str, dist[i])))