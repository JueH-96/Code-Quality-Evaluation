# YOUR CODE HERE
from math import comb

n, q = map(int, input().split())
p = [0] + list(map(int, input().split())) + [n+1]
v = list(map(int, input().split()))

dp = [0] * (q + 1)
dp[0] = 1

for i in range(1, q + 1):
    for j in range(i, 0, -1):
        if v[i-1] >= max(v[:p[j]-1]):
            dp[i] += dp[j-1]
            dp[i] %= 998244353

print(dp[q])