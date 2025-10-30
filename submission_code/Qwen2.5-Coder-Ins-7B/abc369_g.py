import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
for i in range(N-1):
    u = int(data[2*i+1])
    v = int(data[2*i+2])
    l = int(data[2*i+3])
    edges.append((u, v, l))

# Build the tree
from collections import defaultdict
tree = defaultdict(list)
for u, v, l in edges:
    tree[u].append((v, l))
    tree[v].append((u, l))

# Function to find the shortest path from root to all other nodes
def shortest_path(root):
    from heapq import heappop, heappush
    dist = [float('inf')] * (N+1)
    dist[root] = 0
    pq = [(0, root)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, l in tree[u]:
            if dist[u] + l < dist[v]:
                dist[v] = dist[u] + l
                heappush(pq, (dist[v], v))
    return dist

# Calculate shortest paths from root 1
dist = shortest_path(1)

# Function to find the maximum score for given K
def max_score(K):
    if K == 1:
        return 2 * dist[3]
    elif K == 2:
        return 2 * (dist[5] + dist[3])
    else:
        return 2 * (dist[5] + dist[3] + dist[2])

# Print the results for K = 1 to N
for K in range(1, N+1):
    print(max_score(K))