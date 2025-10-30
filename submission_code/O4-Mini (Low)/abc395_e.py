import sys
import threading
import heapq

def main():
    import sys
    input = sys.stdin.readline
    N, M, X = map(int, input().split())
    adj_out = [[] for _ in range(N+1)]
    adj_in  = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj_out[u].append(v)
        adj_in[v].append(u)
    # dist[v][p]: min cost to reach vertex v with parity p (0=normal,1=reversed)
    INF = 10**30
    dist = [[INF, INF] for _ in range(N+1)]
    dist[1][0] = 0
    hq = [(0, 1, 0)]  # (cost, vertex, parity)
    while hq:
        d, v, p = heapq.heappop(hq)
        if d > dist[v][p]:
            continue
        # 1) Flip all edges (toggle parity)
        np = 1 - p
        nd = d + X
        if nd < dist[v][np]:
            dist[v][np] = nd
            heapq.heappush(hq, (nd, v, np))
        # 2) Move along directed edges in current orientation
        if p == 0:
            # use original directions
            for u in adj_out[v]:
                nd = d + 1
                if nd < dist[u][p]:
                    dist[u][p] = nd
                    heapq.heappush(hq, (nd, u, p))
        else:
            # use reversed directions => follow original incoming edges
            for u in adj_in[v]:
                nd = d + 1
                if nd < dist[u][p]:
                    dist[u][p] = nd
                    heapq.heappush(hq, (nd, u, p))
    ans = min(dist[N][0], dist[N][1])
    print(ans)

if __name__ == "__main__":
    main()