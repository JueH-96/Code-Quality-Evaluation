MOD = 998244353
N = 200005

n, m, k = map(int, input().split())
x = [0] * (n + m + 1)
y = [0] * (n + m + 1)
for i in range(1, m + 1):
    x[i], y[i] = map(int, input().split())

dp = [[0] * (n + 1) for _ in range(k + 1)]
dp[0][1] = 1

for i in range(1, k + 1):
    s = [0] * (n + 1)
    for j in range(1, n + 1):
        s[j] = (s[j - 1] + dp[i - 1][j]) % MOD
    for j in range(1, n + 1):
        dp[i][j] = (s[j - 1] - dp[i - 1][j] + MOD) % MOD
    for j in range(1, m + 1):
        dp[i][y[j]] = (dp[i][y[j]] + dp[i - 1][x[j]]) % MOD

print(dp[k][1])