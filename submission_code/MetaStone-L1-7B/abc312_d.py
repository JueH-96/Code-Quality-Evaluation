MOD = 998244353

S = input().strip()
n = len(S)

# Initialize DP table
dp = [[0] * (n + 1) for _ in range(n + 1)]
dp[0][0] = 1

for i in range(n):
    c = S[i]
    for j in range(i + 1):
        if dp[i][j] == 0:
            continue
        if c == '(':
            new_j = j + 1
            if new_j <= i + 1:
                dp[i + 1][new_j] = (dp[i + 1][new_j] + dp[i][j]) % MOD
        elif c == ')':
            new_j = j - 1
            if new_j >= 0:
                dp[i + 1][new_j] = (dp[i + 1][new_j] + dp[i][j]) % MOD
        else:  # '?'
            new_j1 = j + 1
            new_j2 = j - 1
            if new_j1 <= i + 1:
                dp[i + 1][new_j1] = (dp[i + 1][new_j1] + dp[i][j]) % MOD
            if new_j2 >= 0:
                dp[i + 1][new_j2] = (dp[i + 1][new_j2] + dp[i][j]) % MOD

print(dp[n][0] % MOD)