import sys
def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    INF = 1 << 60

    # Initialize distance matrix
    dist = [[INF] * (N+1) for _ in range(N+1)]
    for i in range(N+1):
        dist[i][i] = 0

    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        if dist[u][v] > w:
            dist[u][v] = w

    # Floyd-Warshall algorithm to compute all-pairs shortest paths
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Initialize DP table
    dp = [[INF] * (N+1) for _ in range(1 << N)]
    for u in range(1, N+1):
        mask = 1 << (u-1)
        dp[mask][u] = 0

    # Iterate over all masks
    for mask in range(1 << N):
        for u in range(1, N+1):
            if not (mask & (1 << (u-1))):
                continue
            # Try to go to each v not in mask
            for v in range(1, N+1):
                if mask & (1 << (v-1)):
                    continue
                new_mask = mask | (1 << (v-1))
                if dp[new_mask][v] > dp[mask][u] + dist[u][v]:
                    dp[new_mask][v] = dp[mask][u] + dist[u][v]

    full_mask = (1 << N) - 1
    ans = INF
    for u in range(1, N+1):
        ans = min(ans, dp[full_mask][u])

    if ans == INF:
        print("No")
    else:
        print(ans)

if __name__ == '__main__':
    main()