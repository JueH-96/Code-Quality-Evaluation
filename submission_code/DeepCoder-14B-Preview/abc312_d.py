MOD = 998244353

S = input().strip()
n = len(S)

# Initialize DP table
dp = [[0] * (n + 1) for _ in range(n + 1)]
dp[0][0] = 1

for i in range(n):
    for j in range(n + 1):
        if dp[i][j] == 0:
            continue
        current_char = S[i]
        if current_char == '(':
            new_j = j + 1
            if new_j <= n:
                dp[i+1][new_j] = (dp[i+1][new_j] + dp[i][j]) % MOD
        elif current_char == ')':
            if j >= 1:
                new_j = j - 1
                dp[i+1][new_j] = (dp[i+1][new_j] + dp[i][j]) % MOD
        else:  # '?'
            # Treat as '('
            new_j = j + 1
            if new_j <= n:
                dp[i+1][new_j] = (dp[i+1][new_j] + dp[i][j]) % MOD
            # Treat as ')'
            if j >= 1:
                new_j = j - 1
                dp[i+1][new_j] = (dp[i+1][new_j] + dp[i][j]) % MOD

print(dp[n][0] % MOD)