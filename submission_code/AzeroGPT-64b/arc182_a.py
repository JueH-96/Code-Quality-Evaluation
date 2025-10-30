import sys
N, Q = map(int, input().split())
MOD = 998244353
aa = []
bb = []
for _ in range(Q):
    l, r = map(int, input().split())
    aa.append(l)
    bb.append(r)

P = [0] * (N+1)
V = [0] * (N+1)
for i in range(Q-1, -1, -1):
    P[i] = aa[i]
    V[i] = bb[i]

ans = pow(2, Q, MOD)
dp = [0] * (N+1)
dp[0] = 1
for q in range(Q)[::-1]:
    next_dp = [0] * (N+1)
    p, v = P[q], V[q]
    for i, x in enumerate(dp):
        next_dp[i] += x * ((1 if i < p else 0) + (1 if i >= p and x == v else 0)) % MOD
        next_dp[i] %= MOD
        if i > 0:
            next_dp[i - 1] += x * (1 if i <= p and x == v else 0)
            next_dp[i - 1] %= MOD
    exp = next_dp[0].bit_count()
    ans *= pow(2, exp - 1, MOD)
    ans %= MOD
    dp = next_dp
print(ans)