import sys
from heapq import nlargest

# Read input
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
Q = int(data[1])

# Initialize adjacency list
adj = [[] for _ in range(N + 1)]

# Process queries
queries = []
index = 2
for i in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        u = int(data[index + 1])
        v = int(data[index + 2])
        adj[u].append(v)
        adj[v].append(u)
        index += 3
    elif query_type == 2:
        v = int(data[index + 1])
        k = int(data[index + 2])
        queries.append((v, k, i))
        index += 3

# Process Type 2 queries
results = [-1] * len(queries)
for i, (v, k, query_index) in enumerate(queries):
    connected = set(adj[v])
    if len(connected) >= k:
        results[query_index] = nlargest(k, connected)[-1]

# Output results
for result in results:
    print(result)