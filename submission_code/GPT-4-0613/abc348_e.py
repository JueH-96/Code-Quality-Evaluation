import sys
from heapq import *
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    edges = defaultdict(list)
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    C = list(map(int, sys.stdin.readline().split()))
    C.sort(reverse=True)
    visited = [0]*(N+1)
    q = [1]
    ans = [0]*(N+1)
    i = 0
    while q:
        v = q.pop()
        visited[v] = 1
        ans[v] = C[i]
        i += 1
        for u in edges[v]:
            if visited[u] == 0:
                q.append(u)
    print(sum(C) - max(C))
    print(*[ans[i] for i in range(1, N+1)])

solve()