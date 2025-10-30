t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    dp = [[0]*2 for _ in range(n+1)]
    dp[1][0] = max(0, a[0])
    dp[1][1] = max(0, -a[0])
    for i in range(2, n+1):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + a[i-1])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0] - a[i-1])
    print(max(dp[n]))