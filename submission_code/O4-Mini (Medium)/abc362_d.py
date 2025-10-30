import sys
import threading

def main():
    import sys
    import heapq

    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N, M = map(int, line)
    A = [0] + list(map(int, data.readline().split()))  # 1-based

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, b = map(int, data.readline().split())
        adj[u].append((v, b))
        adj[v].append((u, b))

    INF = 10**40
    dist = [INF] * (N+1)
    dist[1] = A[1]

    # Min-heap of (distance, node)
    hq = [(dist[1], 1)]
    while hq:
        d_u, u = heapq.heappop(hq)
        if d_u != dist[u]:
            continue
        # Relax edges
        for v, w in adj[u]:
            # cost to travel u->v is edge w plus vertex weight A[v]
            nd = d_u + w + A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

    # Output distances from 1 to i for i=2..N
    out = " ".join(str(dist[i]) for i in range(2, N+1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()