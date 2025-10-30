def minimal_coins(prices):
    n = len(prices)
    if n == 0:
        return 0
    dp = [float('inf')] * (n + 2)
    dp[n] = 0
    for i in range(n-1, 0, -1):
        min_cost = float('inf')
        for j in range(max(1, (i) // 2), i):
            if j + j >= i and j <= i - 1:
                cost = prices[j-1] + dp[j + j + 1]
                if cost < min_cost:
                    min_cost = cost
        if min_cost != float('inf'):
            dp[i] = min_cost
        else:
            dp[i] = prices[i-1]
    return dp[1]