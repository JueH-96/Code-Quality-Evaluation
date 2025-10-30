# YOUR CODE HERE
n, d, p = map(int, input().split())
f = list(map(int, input().split()))

dp = [0] * (n + 1)

for i in range(1, n + 1):
    dp[i] = dp[i - 1] + f[i - 1]
    k = (i // d) * d
    while k >= 0:
        dp[i] = min(dp[i], dp[i - k] + (i // d - (k // d)) * p)
        k -= d

print(dp[n])