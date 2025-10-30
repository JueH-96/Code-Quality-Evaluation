# YOUR CODE HERE
from collections import defaultdict

N, Q = map(int, input().split())
graph = defaultdict(set)
isolated = set(range(1, N+1))

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        u, v = query[1], query[2]
        graph[u].add(v)
        graph[v].add(u)
        if u in isolated:
            isolated.remove(u)
        if v in isolated:
            isolated.remove(v)
    elif query[0] == 2:
        v = query[1]
        if v in isolated:
            isolated.remove(v)
        for u in graph[v]:
            graph[u].remove(v)
            if u in isolated:
                isolated.add(u)
            if v in isolated:
                isolated.add(v)
    print(len(isolated))