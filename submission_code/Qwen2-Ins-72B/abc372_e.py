from heapq import heappush, heappop

N, Q = map(int, input().split())
G = [[] for _ in range(N)]
Qs = [tuple(map(int, input().split())) for _ in range(Q)]
hq = [[] for _ in range(N)]
deg = [0] * N
for i, (t, u, v) in enumerate(Qs):
    if t == 1:
        u -= 1
        v -= 1
        G[u].append(v)
        G[v].append(u)
        deg[u] += 1
        deg[v] += 1
        if deg[u] == 1:
            heappush(hq[u], -u)
        if deg[v] == 1:
            heappush(hq[v], -v)
        for v2 in G[u]:
            if v2 == v:
                continue
            heappush(hq[v2], -u)
        for v2 in G[v]:
            if v2 == u:
                continue
            heappush(hq[u], -v)
    else:
        u -= 1
        k = v
        while hq[u] and k > 0:
            k -= 1
            v = -heappop(hq[u])
            if v == u or deg[v] == 0:
                continue
            for v2 in G[v]:
                if v2 == u:
                    continue
                heappush(hq[u], -v)
            deg[v] = 0
        if k > 0:
            print(-1)
        else:
            print(-hq[u][0])