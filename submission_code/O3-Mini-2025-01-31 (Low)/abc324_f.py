def main():
    import sys
    input = sys.stdin.readline
    
    # Read the number of vertices and edges
    N, M = map(int, input().split())
    # Build the graph as an adjacency list: for each u, store (v, b, c)
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        graph[u].append((v, b, c))
    
    # Check feasibility of a ratio r
    def feasible(r):
        # dp[i] will be maximum sum of (b - r*c) from 1 to i
        dp = [-10**18] * (N+1)
        dp[1] = 0  # start with vertex 1
        # Because graph is DAG with u < v, iterate vertices in increasing order
        for u in range(1, N+1):
            if dp[u] == -10**18:
                continue
            for v, b, c in graph[u]:
                candidate = dp[u] + (b - r * c)
                if candidate > dp[v]:
                    dp[v] = candidate
        return dp[N] >= 0

    # Binary search for r in [0, 10000]
    low = 0.0
    high = 10000.0
    eps = 1e-10
    for _ in range(100):  # 100 iterations are more than enough for required precision
        mid = (low + high) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    # low holds the maximum ratio achieving a non-negative sum
    print("{:.16f}".format(low))

if __name__ == '__main__':
    main()