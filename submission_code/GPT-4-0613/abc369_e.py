import sys
from heapq import heappop, heappush

INF = 10**18
N, M = map(int, sys.stdin.readline().split())
edges = [None]*M
G = [[] for _ in range(N)]
for i in range(M):
    u, v, t = map(int, sys.stdin.readline().split())
    u -= 1
    v -= 1
    edges[i] = (u, v, t)
    G[u].append((v, t, i))
    G[v].append((u, t, i))

dist = [[INF]*N for _ in range(N)]
for i in range(N):
    dist[i][i] = 0
    que = [(0, i)]
    while que:
        d, v = heappop(que)
        if dist[i][v] < d:
            continue
        for to, t, _ in G[v]:
            if dist[i][to] > d+t:
                dist[i][to] = d+t
                heappush(que, (d+t, to))

Q = int(sys.stdin.readline())
for _ in range(Q):
    K = int(sys.stdin.readline())
    B = list(map(int, sys.stdin.readline().split()))
    for i in range(K):
        B[i] -= 1
    ans = INF
    for S in range(1<<K):
        for i in range(K):
            if S>>i&1:
                continue
            u, v, t = edges[B[i]]
            tmp = 0
            for j in range(K):
                if S>>j&1:
                    tmp += dist[v][edges[B[j]][1]]+edges[B[j]][2]
                else:
                    tmp += min(dist[u][edges[B[j]][0]]+edges[B[j]][2], dist[v][edges[B[j]][0]]+edges[B[j]][2])
            ans = min(ans, tmp)
    print(ans)