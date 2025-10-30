def solve():
    k = int(input())
    s = input()
    t = input()
    
    n = len(s)
    m = len(t)
    
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    
    dp[0][0] = 0
    
    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)
            if i > 0 and j > 0:
                if s[i-1] == t[j-1]:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1])
                else:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1] + 1)
                    
    if dp[n][m] <= k:
        print("Yes")
    else:
        print("No")

solve()