# YOUR CODE HERE
from collections import defaultdict

N, Q = map(int, input().split())
graph = defaultdict(set)
queries = []

for _ in range(Q):
    query = list(map(int, input().split()))
    queries.append(query)

for query in queries:
    if query[0] == 1:
        u, v = query[1], query[2]
        graph[u].add(v)
        graph[v].add(u)
    else:
        v, k = query[1], query[2]
        connected_vertices = sorted(graph[v], reverse=True)
        if len(connected_vertices) < k:
            print(-1)
        else:
            print(connected_vertices[k-1])