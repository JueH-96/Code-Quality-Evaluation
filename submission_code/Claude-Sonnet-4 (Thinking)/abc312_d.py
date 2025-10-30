s = input().strip()
n = len(s)
MOD = 998244353

# dp[i][j] = number of ways to process first i characters with balance j
max_balance = n
dp = [[0] * (max_balance + 1) for _ in range(n + 1)]
dp[0][0] = 1

for i in range(n):
    for j in range(max_balance + 1):
        if dp[i][j] == 0:
            continue
        
        if s[i] == '(':
            if j + 1 <= max_balance:
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
        elif s[i] == ')':
            if j > 0:
                dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD
        else:  # s[i] == '?'
            # Replace with '('
            if j + 1 <= max_balance:
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
            # Replace with ')'
            if j > 0:
                dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD

print(dp[n][0])