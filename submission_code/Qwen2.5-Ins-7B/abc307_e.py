# YOUR CODE HERE
n, m = map(int, input().split())
mod = 998244353

if n == 1:
    print(m)
    exit()

dp = [0] * (n + 1)
dp[1] = m
dp[2] = m * (m - 1)

for i in range(3, n + 1):
    dp[i] = (dp[i - 1] * (m - 1) + dp[i - 2] * (m - 1)) % mod

print(dp[n])