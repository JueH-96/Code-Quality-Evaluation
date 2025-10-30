def solve():
    k = int(input())
    s = input()
    t = input()

    n = len(s)
    m = len(t)

    dp = [[(k + 1) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(n + 1):
        for j in range(m + 1):
            if i < n and j < m and s[i] == t[j]:
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j])
            
            if i < n:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + 1)
            
            if j < m:
                dp[i][j+1] = min(dp[i][j+1], dp[i][j] + 1)
            
            if i < n and j < m:
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + 1)

    if dp[n][m] <= k:
        print("Yes")
    else:
        print("No")

solve()