import sys
from collections import deque

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
X = [0] * N
Y = [0] * N
for i in range(N):
    X[i] = int(data[index])
    Y[i] = int(data[index + 1])
    index += 2

# Build adjacency list for comparability graph
adj = [[] for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        if ((X[i] < X[j] and Y[i] < Y[j]) or (X[i] > X[j] and Y[i] > Y[j])):
            adj[i].append(j)
            adj[j].append(i)

# Visited array
vis = [False] * N
ans = 1
MOD = 998244353

# Find all connected components and their sizes using BFS
 for i in range(N):
     if not vis[i]:
         # BFS to find size of component
         size = 0
         queue = deque()
         queue.append(i)
         vis[i] = True
         size = 1
         while queue:
             node = queue.popleft()
             for nei in adj[node]:
                 if not vis[nei]:
                     vis[nei] = True
                     queue.append(nei)
                     size += 1  # Increment size for each new node added
         # Calculate f_m for this component size
         if size == 1:
             f_m = 1
         elif size == 2:
             f_m = 3
         else:  # size >= 3
             f_m = (3 * (size - 2) ) % MOD
         # Multiply to ans with modulo
         ans = (ans * f_m) % MOD

# Output the answer
print(ans)