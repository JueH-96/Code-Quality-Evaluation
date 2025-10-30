import sys
from collections import defaultdict

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    graph = defaultdict(set)
    isolated = set(range(1, N+1))

    for _ in range(Q):
        query = sys.stdin.readline().split()

        if query[0] == '1':
            u, v = map(int, query[1:])
            graph[u].add(v)
            graph[v].add(u)
            if u in isolated:
                isolated.remove(u)
            if v in isolated:
                isolated.remove(v)

        elif query[0] == '2':
            v = int(query[1])
            for u in graph[v]:
                graph[u].remove(v)
                if len(graph[u]) == 0:
                    isolated.add(u)
            del graph[v]

        print(len(isolated))

solve()