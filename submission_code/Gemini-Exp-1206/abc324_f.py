def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))

    def check(x):
        dp = [-float('inf')] * (n + 1)
        dp[1] = 0
        for u in range(1, n):
            if dp[u] == -float('inf'):
                continue
            for edge in edges:
                if edge[0] == u:
                    dp[edge[1]] = max(dp[edge[1]], dp[u] + edge[2] - x * edge[3])
        return dp[n] >= 0

    left, right = 0, 10000
    for _ in range(100):
        mid = (left + right) / 2
        if check(mid):
            left = mid
        else:
            right = mid
    
    print(f"{left:.15f}")

solve()