import sys
from heapq import heappop, heappush

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, t = map(int, sys.stdin.readline().split())
        graph[u].append((v, t))
        graph[v].append((u, t))

    Q = int(sys.stdin.readline())
    for _ in range(Q):
        bridges = list(map(int, sys.stdin.readline().split()))[1:]
        dist = [float('inf')] * (N + 1)
        dist[1] = 0
        pq = [(0, 1)]
        while pq:
            d, u = heappop(pq)
            if d != dist[u]: continue
            for v, t in graph[u]:
                if dist[v] > d + t and (u < v or (u == v and t in bridges)):
                    dist[v] = d + t
                    heappush(pq, (dist[v], v))
        print(dist[N])

solve()