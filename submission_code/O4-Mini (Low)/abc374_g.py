#!/usr/bin/env python3
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    # Build bipartite graph: left side edges 0..N-1, right side edges 0..N-1
    # add edge i->j if S[i][1] == S[j][0]
    adj = [[] for _ in range(N)]
    # precompute list of indices by starting letter
    start_bucket = {}
    for j, sj in enumerate(S):
        start_bucket.setdefault(sj[0], []).append(j)
    for i, si in enumerate(S):
        c = si[1]
        if c in start_bucket:
            adj[i].extend(start_bucket[c])
    # Hopcroft-Karp
    from collections import deque
    INF = 10**9
    # pairU[i]= matched v or -1; pairV[j]= matched u or -1
    pairU = [-1]*N
    pairV = [-1]*N
    dist = [0]*N
    def bfs():
        dq = deque()
        for u in range(N):
            if pairU[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = INF
        d_inf = INF
        while dq:
            u = dq.popleft()
            if dist[u] < d_inf:
                for v in adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        d_inf = dist[u] + 1
                    else:
                        if dist[pu] == INF:
                            dist[pu] = dist[u] + 1
                            dq.append(pu)
        return d_inf != INF

    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pairV[v] == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(N):
            if pairU[u] == -1:
                if dfs(u):
                    matching += 1

    # Minimal path cover of the edge-graph = N - matching
    print(N - matching)

if __name__ == "__main__":
    main()