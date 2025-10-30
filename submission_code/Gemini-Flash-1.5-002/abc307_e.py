# YOUR CODE HERE
MOD = 998244353

N, M = map(int, input().split())

if M == 1:
    print(0)
else:
    dp = [0] * (N + 1)
    dp[0] = 1
    dp[1] = M
    for i in range(2, N + 1):
        dp[i] = (M - 1) * (dp[i - 1] + dp[i - 2]) % MOD
    print(dp[N])