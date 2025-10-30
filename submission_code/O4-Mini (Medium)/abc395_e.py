import sys
import threading
def main():
    import sys
    import heapq

    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0])
    M = int(line[1])
    X = int(line[2])
    # Build adjacency lists for original and reversed edges
    edges_out = [[] for _ in range(N)]
    edges_rev = [[] for _ in range(N)]
    for _ in range(M):
        u,v = data.readline().split()
        u = int(u)-1
        v = int(v)-1
        edges_out[u].append(v)
        edges_rev[v].append(u)

    INF = 10**40
    # dist0[v]: min cost to reach v with even number of reversals (parity 0)
    # dist1[v]: min cost to reach v with odd number of reversals (parity 1)
    dist0 = [INF]*N
    dist1 = [INF]*N
    dist0[0] = 0

    # Min-heap of (cost, vertex, parity)
    hq = [(0, 0, 0)]  # start at vertex 0, parity 0, cost 0

    while hq:
        d, v, p = heapq.heappop(hq)
        # Skip if this is an outdated entry
        if p == 0:
            if d != dist0[v]:
                continue
        else:
            if d != dist1[v]:
                continue

        # 1) Move along edges in current orientation
        if p == 0:
            # original orientation: use edges_out
            nd = d + 1
            for u in edges_out[v]:
                if nd < dist0[u]:
                    dist0[u] = nd
                    heapq.heappush(hq, (nd, u, 0))
        else:
            # reversed orientation: use edges_rev
            nd = d + 1
            for u in edges_rev[v]:
                if nd < dist1[u]:
                    dist1[u] = nd
                    heapq.heappush(hq, (nd, u, 1))

        # 2) Reverse all edges (flip parity) at cost X
        nd = d + X
        if p == 0:
            if nd < dist1[v]:
                dist1[v] = nd
                heapq.heappush(hq, (nd, v, 1))
        else:
            if nd < dist0[v]:
                dist0[v] = nd
                heapq.heappush(hq, (nd, v, 0))

    # We can finish at vertex N-1 in either parity
    ans = dist0[N-1] if dist0[N-1] < dist1[N-1] else dist1[N-1]
    print(ans)

if __name__ == "__main__":
    main()