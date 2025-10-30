MOD = 998244353
S = input().strip()
N = len(S)
S = '?' + S + '?'
dp = [[0]*(N+2) for _ in range(N+2)]
dp[0][0] = 1
for i in range(1, N+2):
    for j in range(N+2):
        if S[i] != '(':
            if j >= 1:
                dp[i][j-1] += dp[i-1][j]
                dp[i][j-1] %= MOD
        if S[i] != ')':
            dp[i][j+1] += dp[i-1][j]
            dp[i][j+1] %= MOD
print(dp[N+1][0])