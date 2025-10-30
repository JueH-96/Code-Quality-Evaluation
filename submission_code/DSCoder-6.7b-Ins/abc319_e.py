import sys
from heapq import heappop, heappush

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    P = [0] + list(map(int, sys.stdin.readline().split()))
    T = [0] + list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    q = [int(sys.stdin.readline()) for _ in range(Q)]

    INF = 10**18
    dist = [INF] * (N+2)
    dist[1] = X
    hq = [(X, 1)]
    while hq:
        d, u = heappop(hq)
        if d != dist[u]:
            continue
        for v in range(u+1, N+2):
            t = d + T[u]
            if t < dist[v]:
                dist[v] = t
                heappush(hq, (t, v))
            t += P[u]
            if t < dist[v]:
                dist[v] = t
                heappush(hq, (t, v))
    dist[N+1] = Y
    heappush(hq, (Y, N+1))

    for d, u in hq:
        if d != dist[u]:
            continue
        for v in range(u-1, 0, -1):
            t = d - T[u-1]
            if t < dist[v]:
                dist[v] = t
                heappush(hq, (t, v))

    for qi in q:
        print(next(d for d in dist if d >= qi))

solve()