def solve():
    n, k, x = map(int, input().split())
    t = list(map(int, input().split()))

    dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, i + 1):
            for p in range(1, min(k, j) + 1):
                ship_day = max(t[i-1], t[i-p] + (j - p) * x if i-p>=0 else t[i-1])
                
                dissatisfaction = 0
                for q in range(p):
                    dissatisfaction += ship_day - t[i - p + q]
                
                dp[i][j] = min(dp[i][j], dp[i - p][j - p] + dissatisfaction)

    print(min(dp[n]))

solve()