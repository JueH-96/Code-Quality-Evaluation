MOD = 998244353

K = int(input())
C = list(map(int, input().split()))

max_fact = K
fact = [1] * (max_fact + 1)
for i in range(1, max_fact + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact = [1] * (max_fact + 1)
inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
for i in range(max_fact-1, -1, -1):
    inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

dp = [0] * (K + 1)
dp[0] = 1

for c in C:
    new_dp = [0] * (K + 1)
    for a in range(K + 1):
        if dp[a] == 0:
            continue
        max_t = min(c, K - a)
        for t in range(0, max_t + 1):
            new_dp[a + t] = (new_dp[a + t] + dp[a] * inv_fact[t]) % MOD
    dp = new_dp

result = 0
for k in range(1, K + 1):
    result = (result + dp[k] * fact[k]) % MOD

print(result)