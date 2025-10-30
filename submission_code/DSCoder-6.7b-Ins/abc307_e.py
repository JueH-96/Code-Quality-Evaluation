# YOUR CODE HERE

def solve(N, M):
    MOD = 998244353
    dp = [[0]*M for _ in range(N+1)]
    for i in range(M):
        dp[1][i] = 1
    for i in range(2, N+1):
        dp2 = [0]*M
        for j in range(M):
            dp2[j] = (dp2[j] + dp[i-1][(j-1)%M]) % MOD
            dp2[j] = (dp2[j] + dp[i-1][(j+1)%M]) % MOD
        dp[i] = dp2
    return sum(dp[-1]) % MOD

N, M = map(int, input().split())
print(solve(N, M))