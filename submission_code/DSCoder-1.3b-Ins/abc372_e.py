import sys
from collections import defaultdict

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    graph = defaultdict(list)
    for _ in range(Q):
        query_type, u, v = map(int, sys.stdin.readline().split())
        if query_type == 1:
            graph[u].append(v)
            graph[v].append(u)
        else:
            graph[v].sort(reverse=True)
            print(graph[v][0] if len(graph[v]) >= u else -1)

solve()