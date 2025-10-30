import sys
from collections import deque

def max_distance_from_source(adj, start):
    dist = [-1] * len(adj)
    dist[start] = 0
    queue = deque([start])
    max_dist = 0
    while queue:
        node = queue.popleft()
        for nei in adj[node]:
            if dist[nei] == -1:
                dist[nei] = dist[node] + 1
                queue.append(nei)
                if dist[nei] > max_dist:
                    max_dist = dist[nei]
    return max_dist

# Read input
data = sys.stdin.read().split()
index = 0
N1 = int(data[index])
index += 1
N2 = int(data[index])
index += 1
M = int(data[index])
index += 1

N = N1 + N2  # Total number of nodes
adj = [[] for _ in range(N + 1)]  # Adjacency list, indexed from 1 to N

# Read M edges and build adjacency list
for _ in range(M):
    a = int(data[index])
    index += 1
    b = int(data[index])
    index += 1
    adj[a].append(b)
    adj[b].append(a)

# Compute maximum distance in first component from vertex 1
maxA = max_distance_from_source(adj, 1)

# Compute maximum distance in second component from vertex N1 + N2
T = N1 + N2
maxB = max_distance_from_source(adj, T)

# The answer is maxA + maxB + 1
answer = maxA + maxB + 1

# Output the answer
print(answer)