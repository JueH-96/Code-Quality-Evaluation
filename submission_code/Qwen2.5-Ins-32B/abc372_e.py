# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
sys.setrecursionlimit(2 * 10**5 + 100)

N, Q, *queries = map(int, input().split())

graph = defaultdict(set)

idx = 0
for _ in range(Q):
    query_type = queries[idx]
    idx += 1
    if query_type == 1:
        u, v = queries[idx:idx+2]
        idx += 2
        graph[u].add(v)
        graph[v].add(u)
    else:
        v, k = queries[idx:idx+2]
        idx += 2
        connected_vertices = sorted(graph[v], reverse=True)
        if len(connected_vertices) < k:
            print(-1)
        else:
            print(connected_vertices[k-1])