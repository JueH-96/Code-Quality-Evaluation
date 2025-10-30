# YOUR CODE HERE
K = int(input())
C = list(map(int, input().split()))
MOD = 998244353

dp = [[0] * (K + 1) for _ in range(27)]
dp[0][0] = 1

for i in range(26):
    for j in range(K + 1):
        for k in range(C[i] + 1):
            if j + k <= K:
                dp[i + 1][j + k] = (dp[i + 1][j + k] + dp[i][j]) % MOD

ans = 0
for i in range(1, K + 1):
    ans = (ans + dp[26][i]) % MOD

print(ans)