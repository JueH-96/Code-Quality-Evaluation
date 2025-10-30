import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N, M = map(int, input().split())
    edges = []
    G = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b, c = map(int, input().split())
        edges.append((a, b, c))
        G[a].append((b, c))
        G[b].append((a, c))

    import heapq

    INF = 10**30
    # Dijkstra from 1
    dist1 = [INF]*(N+1)
    dist1[1] = 0
    cnt1_mod1 = [0]*(N+1)
    cnt1_mod2 = [0]*(N+1)
    cnt1_mod1[1] = 1
    cnt1_mod2[1] = 1
    MOD1 = 10**9+7
    MOD2 = 10**9+9
    pq = [(0,1)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>dist1[u]: continue
        for v,w in G[u]:
            nd = d + w
            if nd < dist1[v]:
                dist1[v] = nd
                cnt1_mod1[v] = cnt1_mod1[u]
                cnt1_mod2[v] = cnt1_mod2[u]
                heapq.heappush(pq, (nd, v))
            elif nd == dist1[v]:
                cnt1_mod1[v] = (cnt1_mod1[v] + cnt1_mod1[u]) % MOD1
                cnt1_mod2[v] = (cnt1_mod2[v] + cnt1_mod2[u]) % MOD2

    # Dijkstra from N
    distN = [INF]*(N+1)
    distN[N] = 0
    cntN_mod1 = [0]*(N+1)
    cntN_mod2 = [0]*(N+1)
    cntN_mod1[N] = 1
    cntN_mod2[N] = 1
    pq = [(0,N)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>distN[u]: continue
        for v,w in G[u]:
            nd = d + w
            if nd < distN[v]:
                distN[v] = nd
                cntN_mod1[v] = cntN_mod1[u]
                cntN_mod2[v] = cntN_mod2[u]
                heapq.heappush(pq, (nd, v))
            elif nd == distN[v]:
                cntN_mod1[v] = (cntN_mod1[v] + cntN_mod1[u]) % MOD1
                cntN_mod2[v] = (cntN_mod2[v] + cntN_mod2[u]) % MOD2

    D = dist1[N]
    total1 = cnt1_mod1[N]
    total2 = cnt1_mod2[N]

    out = []
    for (u, v, c) in edges:
        paths1 = 0
        paths2 = 0
        # u->v direction
        if dist1[u] + c + distN[v] == D:
            paths1 = (paths1 + cnt1_mod1[u] * cntN_mod1[v]) % MOD1
            paths2 = (paths2 + cnt1_mod2[u] * cntN_mod2[v]) % MOD2
        # v->u direction
        if dist1[v] + c + distN[u] == D:
            paths1 = (paths1 + cnt1_mod1[v] * cntN_mod1[u]) % MOD1
            paths2 = (paths2 + cnt1_mod2[v] * cntN_mod2[u]) % MOD2

        # If it's on no shortest path, removal doesn't change the shortest distance
        if paths1 == 0 and paths2 == 0:
            out.append("No")
        else:
            # If it accounts for all shortest paths, removing it destroys all shortest paths
            if paths1 == total1 and paths2 == total2:
                out.append("Yes")
            else:
                out.append("No")

    print("
".join(out))

if __name__ == "__main__":
    main()