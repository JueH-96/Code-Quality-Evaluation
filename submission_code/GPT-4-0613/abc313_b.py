import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    G = defaultdict(list)
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        G[a].append(b)
        G[b].append(a)
    visited = [False] * (N + 1)
    def dfs(v):
        visited[v] = True
        for u in G[v]:
            if not visited[u]:
                dfs(u)
    dfs(1)
    if all(visited[1:]):
        print(1)
    else:
        print(-1)

solve()