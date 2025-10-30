import sys
import heapq


def main() -> None:
    input_data = sys.stdin.buffer.readline

    # read N, M
    n, m = map(int, input_data().split())

    # read vertex weights
    A = list(map(int, input_data().split()))

    # build adjacency list
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v, b = map(int, input_data().split())
        u -= 1  # to 0-based index
        v -= 1

        # travelling u -> v means paying edge-weight + vertex-weight of v
        adj[u].append((v, b + A[v]))
        # travelling v -> u means paying edge-weight + vertex-weight of u
        adj[v].append((u, b + A[u]))

    # Dijkstra
    INF = 10 ** 20
    dist = [INF] * n
    dist[0] = A[0]                       # cost of staying at vertex 1
    pq = [(dist[0], 0)]                  # (distance, vertex)

    while pq:
        d, v = heapq.heappop(pq)
        if d != dist[v]:
            continue
        for nxt, w in adj[v]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))

    # output answers for vertices 2..N
    print(' '.join(str(dist[i]) for i in range(1, n)))


if __name__ == "__main__":
    main()