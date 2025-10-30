#!/usr/bin/env python3
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N = int(input())
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for _ in range(N-1):
        u,v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    # 1) find final leaves (deg==1)
    leaves = [u for u in range(1, N+1) if deg[u] == 1]
    # 2) count c1[v] = number of deg1 neighbors
    c1 = [0]*(N+1)
    for u in leaves:
        # u is deg1, its only neighbor is a center of some star
        for v in adj[u]:
            c1[v] += 1
    # 3) initial centers: those with c1>0
    centers = set(u for u in range(1, N+1) if c1[u] > 0)
    # 4) count deg2 and deg>=3 nodes
    d2 = 0
    c3 = 0
    for u in range(1, N+1):
        if deg[u] == 2:
            d2 += 1
        elif deg[u] >= 3:
            c3 += 1
    # 5) compute number of L=2 centers total
    # formula: c2 = (d2 + 2 - 2*c3) / 3
    # it must be integral
    num_c2 = (d2 + 2 - 2*c3) // 3
    # of those, some are already in centers with deg==2 and c1>0
    already_c2 = 0
    for u in centers:
        if deg[u] == 2:
            already_c2 += 1
    need_more = num_c2 - already_c2
    # 6) if need_more > 0, we must pick that many ambiguous deg2 nodes
    # ambiguous candidates = nodes with deg==2 and c1==0
    cand = []
    if need_more > 0:
        # BFS from all leaves to get dist to nearest leaf
        from collections import deque
        dist = [-1]*(N+1)
        dq = deque()
        for u in leaves:
            dist[u] = 0
            dq.append(u)
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        # collect deg2 & c1==0 nodes
        for u in range(1, N+1):
            if deg[u] == 2 and c1[u] == 0:
                cand.append((dist[u], u))
        # sort descending by dist, pick top need_more
        cand.sort(reverse=True)
        for i in range(need_more):
            centers.add(cand[i][1])
    # 7) now centers is full set of star-centers
    #    the level L for center u is just deg[u]
    L = [deg[u] for u in centers]
    L.sort()
    # output
    print(" ".join(str(x) for x in L))

if __name__ == "__main__":
    main()