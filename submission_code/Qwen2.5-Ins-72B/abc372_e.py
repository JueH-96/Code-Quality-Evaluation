# YOUR CODE HERE
import sys
from collections import defaultdict

N, Q = map(int, input().split())
graph = defaultdict(set)

def add_edge(u, v):
    graph[u].add(v)
    graph[v].add(u)

def kth_largest(v, k):
    connected = sorted(graph[v], reverse=True)
    if len(connected) < k:
        return -1
    return connected[k-1]

results = []
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1], query[2]
        add_edge(u, v)
    elif query[0] == 2:
        v, k = query[1], query[2]
        results.append(kth_largest(v, k))

for result in results:
    print(result)