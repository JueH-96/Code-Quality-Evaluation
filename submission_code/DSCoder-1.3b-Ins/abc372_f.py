def solve():
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    Y = list(map(int, input().split()))
    mod = 998244353
    dp = [[0]*(K+2) for _ in range(N+1)]
    dp[1][0] = 1
    for i in range(1, N+1):
        for j in range(K+1):
            dp[i][j+1] = (dp[i][j+1] + dp[i-1][j]*(j+1)) % mod
            if j:
                dp[i][j] = (dp[i][j] + dp[i-1][j-1]*(j)) % mod
    ans = 0
    for i in range(1, M+1):
        ans = (ans + dp[X[i-1]][K]) % mod
    print(ans)

solve()