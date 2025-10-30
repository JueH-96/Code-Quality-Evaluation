def solve():
    K = int(input().strip())
    dp = [[0]*11 for _ in range(11)]
    for i in range(10):
        dp[i][1] = 10 - i
    for i in range(2, 11):
        for j in range(10):
            dp[j][i] = dp[j][i-1] + dp[j+1][i]
    ans = ''
    d = 1
    while dp[0][d] < K:
        d += 1
    for _ in range(d):
        for i in range(10):
            if dp[i][d] >= K:
                ans += str(i)
                K -= dp[i+1][d]
                d -= 1
                break
    print(ans)

solve()