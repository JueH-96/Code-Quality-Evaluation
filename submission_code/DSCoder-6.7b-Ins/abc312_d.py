MOD = 998244353
N = 3005
dp = [[0]*N for _ in range(N)]
dp[0][0] = 1
for i in range(1, N):
    dp[i][0] = dp[i-1][1]
    for j in range(1, i+1):
        dp[i][j] = (dp[i-1][j-1] + dp[i-1][j]) % MOD

s = input().strip()
n = len(s)
q = s.count('?')
ans = dp[n][q]

c = [0]*N
c[0] = 1
for i in range(n):
    if s[i] == '(':
        for j in range(q, 0, -1):
            c[j] = (c[j] + c[j-1]) % MOD
    elif s[i] == ')':
        for j in range(q, 0, -1):
            c[j] = (c[j] + c[j]) % MOD
    else:
        for j in range(q, 0, -1):
            c[j] = (c[j]*2 + c[j-1]) % MOD

print(ans*c[q]%MOD)