import sys
import threading

def main():
    import sys
    import heapq
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A_list = list(map(int, input().split()))
    # make A 1-indexed
    A = [0] + A_list

    # build adjacency list: for each node store (neighbor, edge_weight)
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, b = map(int, input().split())
        adj[u].append((v, b))
        adj[v].append((u, b))

    # Dijkstra's algorithm
    INF = 10**40
    dist = [INF] * (N+1)
    # distance to node 1 includes its vertex weight A[1]
    dist[1] = A[1]

    hq = [(dist[1], 1)]
    while hq:
        d, u = heapq.heappop(hq)
        if d != dist[u]:
            continue
        # relax edges
        for v, bw in adj[u]:
            nd = d + bw + A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

    # output answers for vertices 2..N
    out = " ".join(str(dist[i]) for i in range(2, N+1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()