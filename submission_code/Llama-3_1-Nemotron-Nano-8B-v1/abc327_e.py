import math

n = int(input())
P = list(map(int, input().split()))

max_k = n
dp = [-float('inf')] * (max_k + 1)
dp[0] = 0.0

for i in range(1, n + 1):
    p = P[i - 1]
    for k in range(n, 0, -1):
        if dp[k - 1] != -float('inf'):
            dp[k] = max(dp[k], dp[k - 1] * 0.9 + p)

max_R = -float('inf')

for k in range(1, n + 1):
    if dp[k] == -float('inf'):
        continue
    denominator = 10 * (1 - 0.9 ** k)
    sum_S = dp[k]
    R = sum_S / denominator - 1200 / math.sqrt(k)
    if R > max_R:
        max_R = R

print("{0:.15f}".format(max_R))