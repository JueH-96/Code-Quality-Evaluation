import sys
import heapq

def main():
    input = sys.stdin.readline
    N = int(input())
    # Read A, B, X for stages 1 through N-1
    A = [0] * (N + 1)
    B = [0] * (N + 1)
    X = [0] * (N + 1)
    for i in range(1, N):
        ai, bi, xi = map(int, input().split())
        A[i] = ai
        B[i] = bi
        X[i] = xi

    # Dijkstra's algorithm
    INF = 10**30
    dist = [INF] * (N + 1)
    dist[1] = 0
    hq = [(0, 1)]  # (distance, node)

    while hq:
        d, u = heapq.heappop(hq)
        if d > dist[u]:
            continue
        # Once we pop stage N with the minimal distance, we can stop
        if u == N:
            break
        # From stage u, we have up to two moves if u < N
        if u < N:
            # Move to u+1 with cost A[u]
            v = u + 1
            nd = d + A[u]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

            # Jump to X[u] with cost B[u]
            v2 = X[u]
            nd2 = d + B[u]
            if nd2 < dist[v2]:
                dist[v2] = nd2
                heapq.heappush(hq, (nd2, v2))

    # Print the minimum time to be able to play stage N
    print(dist[N])

if __name__ == "__main__":
    main()