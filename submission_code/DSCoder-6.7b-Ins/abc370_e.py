import sys

MOD = 998244353
N, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Precompute factorials
fact = [1] * (N + 1)
for i in range(1, N + 1):
    fact[i] = (fact[i - 1] * i) % MOD

# Precompute inverse factorials
inv_fact = [1] * (N + 1)
inv_fact[N] = pow(fact[N], MOD - 2, MOD)
for i in range(N, 0, -1):
    inv_fact[i - 1] = (inv_fact[i] * i) % MOD

# Precompute binomial coefficients
binom = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(N + 1):
    binom[i][0] = 1
    for j in range(1, i + 1):
        binom[i][j] = (binom[i - 1][j - 1] + binom[i - 1][j]) % MOD

# Count divisions with no subsequence summing to K
dp = [0] * (N + 1)
dp[0] = 1
for i in range(1, N + 1):
    dp2 = dp[:]
    for j in range(i - 1, -1, -1):
        dp2[j] = (dp2[j + 1] + dp[j] * fact[i - 1] * inv_fact[j] % MOD * inv_fact[i - 1 - j]) % MOD
    dp = dp2
    if i % 2 == 1:
        for j in range(N - i + 1):
            dp[j + 1] = (dp[j + 1] + dp[j] * fact[i - 1] * inv_fact[j] % MOD * inv_fact[i - 1 - j]) % MOD

# Count divisions with no subsequence summing to K - sum(A)
dp = [0] * (N + 1)
dp[0] = 1
for i in range(1, N + 1):
    dp2 = dp[:]
    for j in range(i - 1, -1, -1):
        dp2[j] = (dp2[j + 1] + dp[j] * fact[i - 1] * inv_fact[j] % MOD * inv_fact[i - 1 - j]) % MOD
    dp = dp2
    if i % 2 == 1:
        for j in range(N - i + 1):
            dp[j + 1] = (dp[j + 1] + dp[j] * fact[i - 1] * inv_fact[j] % MOD * inv_fact[i - 1 - j]) % MOD

# Print the count modulo 998244353
print((dp[N] - dp[N - 1] if K <= 0 else dp[N]) % MOD)