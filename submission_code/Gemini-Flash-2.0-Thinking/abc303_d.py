def solve():
    x, y, z = map(int, input().split())
    s = input()
    n = len(s)

    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    dp[0][1] = z

    for i in range(1, n + 1):
        char = s[i - 1]

        # Calculate dp[i][0] (Caps Lock off after producing s[:i])
        cost_from_off = dp[i - 1][0] + (x if char == 'a' else y)
        cost_from_on = dp[i - 1][1] + (y + z if char == 'a' else x + z)
        dp[i][0] = min(cost_from_off, cost_from_on)

        # Calculate dp[i][1] (Caps Lock on after producing s[:i])
        cost_from_off = dp[i - 1][0] + (x + z if char == 'a' else y + z)
        cost_from_on = dp[i - 1][1] + (y if char == 'a' else x)
        dp[i][1] = min(cost_from_off, cost_from_on)

    print(min(dp[n][0], dp[n][1]))

# YOUR CODE HERE
solve()