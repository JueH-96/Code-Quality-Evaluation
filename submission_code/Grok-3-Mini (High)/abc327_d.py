import sys
from collections import deque

def bfs(adj, color, start):
    queue = deque([start])
    color[start] = 0
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if color[v] == -1:
                color[v] = 1 - color[u]
                queue.append(v)
            elif color[v] == color[u]:
                return False
    return True

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
A = list(map(int, data[index:index + M]))
index += M
B = list(map(int, data[index:index + M]))

# Check for self-loops
for i in range(M):
    if A[i] == B[i]:
        print("No")
        sys.exit()

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for i in range(M):
    adj[A[i]].append(B[i])
    adj[B[i]].append(A[i])

# Remove duplicate edges in adjacency list
for u in range(1, N + 1):
    adj[u] = list(set(adj[u]))

# Color array initialization
color = [-1] * (N + 1)

# Check bipartiteness for each component
for node in range(1, N + 1):
    if color[node] == -1:
        if not bfs(adj, color, node):
            print("No")
            sys.exit()

# If no conflicts, print Yes
print("Yes")