# YOUR CODE HERE
mod = 998244353

S = input().strip()
N = len(S)

dp = [ [0]*(N+2) for _ in range(N+1) ]
dp[0][0] = 1

for i in range(N):
    ch = S[i]
    for b in range(N+1):
        if dp[i][b] == 0:
            continue
        # Try to add '('
        if ch == '(' or ch == '?':
            dp[i+1][b+1] = (dp[i+1][b+1] + dp[i][b]) % mod
        # Try to add ')'
        if (ch == ')' or ch == '?') and b > 0:
            dp[i+1][b-1] = (dp[i+1][b-1] + dp[i][b]) % mod

print(dp[N][0] % mod)