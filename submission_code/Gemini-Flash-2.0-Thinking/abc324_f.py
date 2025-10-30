def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u - 1, v - 1, b, c))

    def check(lambda_val):
        dp = [-float('inf')] * n
        dp[0] = 0
        for i in range(n):
            for u, v, b, c in edges:
                if u == i and dp[u] != -float('inf'):
                    weight = b - lambda_val * c
                    dp[v] = max(dp[v], dp[u] + weight)
        return dp[n - 1] >= 0

    low = 0.0
    high = 10000.0  # A reasonable upper bound
    for _ in range(100):  # Number of iterations for desired precision
        mid = (low + high) / 2
        if check(mid):
            low = mid
        else:
            high = mid
    print(f"{low:.12f}")

solve()