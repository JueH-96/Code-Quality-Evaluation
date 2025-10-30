import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.buffer.read().split())
    N  = int(next(it))
    M  = int(next(it))
    X  = int(next(it))

    # index: (parity * N) + (vertex-1)           vertices are 1-based in input
    size = 2 * N
    adj  = [[] for _ in range(size)]

    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        # walk edges, cost = 1
        adj[u].append((v, 1))           # original orientation (parity 0)
        adj[v + N].append((u + N, 1))   # reversed orientation (parity 1)

    # flip edges, cost = X
    for v in range(N):
        adj[v].append((v + N, X))
        adj[v + N].append((v, X))

    INF = 10 ** 30
    dist = [INF] * size
    start = 0            # vertex 1, parity 0
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, v = heapq.heappop(pq)
        if d != dist[v]:
            continue
        if v == N - 1 or v == N - 1 + N:   # reached vertex N (any parity)
            print(d)
            return
        for nxt, w in adj[v]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))

if __name__ == "__main__":
    main()