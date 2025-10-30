def solve():
    s = input()
    n = len(s)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        # Pressing a single digit button
        dp[i] = min(dp[i], dp[i-1] + 1)

        # Pressing the '00' button
        if i >= 2 and s[i-2:i] == "00":
            dp[i] = min(dp[i], dp[i-2] + 1)

    print(dp[n])

solve()