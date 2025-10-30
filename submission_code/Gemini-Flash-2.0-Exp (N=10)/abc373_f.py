def solve():
    n, w = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    dp = [[0 for _ in range(w + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight, value = items[i - 1]
        for j in range(w + 1):
            dp[i][j] = dp[i-1][j]
            for k in range(1, w // weight + 1):
                if j >= k * weight:
                    happiness = k * value - k * k
                    dp[i][j] = max(dp[i][j], dp[i-1][j - k * weight] + happiness)
    
    print(dp[n][w])

solve()