import math

mod = 998244353

N, X = map(int, input().split())
T = list(map(int, input().split()))
T1 = T[0]
all_T = T

inv_N = pow(N, mod-2, mod)
max_s = X
dp = [0] * (max_s + 1)
dp[0] = inv_N  # 1/N

for s in range(1, max_s + 1):
    for t in all_T:
        if t <= s:
            dp[s] = (dp[s] + dp[s - t] * inv_N) % mod

a = max(0, math.ceil(X + 0.5 - T1))
valid_s = range(a, X + 1)
sum_prob = 0
for s in valid_s:
    sum_prob = (sum_prob + dp[s]) % mod

print(sum_prob)