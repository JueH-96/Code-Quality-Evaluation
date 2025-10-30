import sys
from heapq import heappop, heappush

def solve():
    N, M = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    G = [[] for _ in range(N)]
    RG = [[] for _ in range(N)]
    deg = [0]*N
    total = 0
    for u, v, w in edges:
        u -= 1
        v -= 1
        total += w
        if w > 0:
            G[u].append((v, w))
            RG[v].append((u, w))
            deg[v] += 1
        else:
            G[v].append((u, -w))
            RG[u].append((v, -w))
            deg[u] += 1
    heap = [(0, v) for v in range(N) if deg[v] == 0]
    dist = [0]*N
    while heap:
        d, v = heappop(heap)
        if dist[v] < d:
            continue
        for to, w in G[v]:
            if dist[to] > d + w:
                dist[to] = d + w
                heappush(heap, (dist[to], to))
    return total - max(dist) if max(deg) > 0 else total

print(solve())