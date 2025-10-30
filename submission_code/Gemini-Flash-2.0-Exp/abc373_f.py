def solve():
    n, w = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    dp = [[0] * (w + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wi, vi = items[i-1]
        for j in range(w + 1):
            dp[i][j] = dp[i-1][j]
            for k in range(1, j // wi + 1):
                if j >= k * wi:
                    dp[i][j] = max(dp[i][j], dp[i-1][j - k * wi] + k * vi - k * k)
    
    print(dp[n][w])

solve()