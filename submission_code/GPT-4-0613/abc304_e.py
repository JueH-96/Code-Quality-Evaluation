import sys
from collections import defaultdict

def add_edge(u, v):
    graph[u].append(v)
    graph[v].append(u)

def dfs(v, p):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs(u, v)

N, M = map(int, sys.stdin.readline().split())
graph = defaultdict(list)
visited = [False]*(N+1)

for _ in range(M):
    u, v = map(int, sys.stdin.readline().split())
    add_edge(u, v)

K = int(sys.stdin.readline())
XY = [list(map(int, sys.stdin.readline().split())) for _ in range(K)]

Q = int(sys.stdin.readline())
PQ = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

for i in range(Q):
    p, q = PQ[i]
    add_edge(p, q)
    visited = [False]*(N+1)
    dfs(p, -1)
    x, y = XY[i]
    if visited[x] and visited[y]:
        print("No")
    else:
        print("Yes")