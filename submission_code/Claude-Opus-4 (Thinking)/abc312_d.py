MOD = 998244353

S = input().strip()
n = len(S)

# dp[i][j] = number of ways to process first i characters with balance j
# balance = number of '(' minus number of ')'
dp = [[0] * (n + 1) for _ in range(n + 1)]
dp[0][0] = 1

for i in range(n):
    for j in range(n + 1):
        if dp[i][j] == 0:
            continue
        
        if S[i] == '(':
            if j + 1 <= n:
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
        elif S[i] == ')':
            if j > 0:
                dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD
        else:  # S[i] == '?'
            # Replace with '('
            if j + 1 <= n:
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
            # Replace with ')'
            if j > 0:
                dp[i + 1][j - 1] = (dp[i + 1][j - 1] + dp[i][j]) % MOD

print(dp[n][0])