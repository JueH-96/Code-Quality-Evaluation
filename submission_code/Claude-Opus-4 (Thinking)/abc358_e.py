# YOUR CODE HERE
K = int(input())
C = list(map(int, input().split()))

MOD = 998244353

# Precompute factorials and inverse factorials for combinations
fact = [1] * (K + 1)
for i in range(1, K + 1):
    fact[i] = fact[i-1] * i % MOD

inv_fact = [1] * (K + 1)
inv_fact[K] = pow(fact[K], MOD - 2, MOD)
for i in range(K - 1, -1, -1):
    inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

def comb(n, k):
    if k > n or k < 0:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

# dp[i][j] = number of strings of length i using the first j letters
dp = [[0] * 27 for _ in range(K + 1)]

# Base case: empty string
for j in range(27):
    dp[0][j] = 1

# Fill the dp table
for j in range(1, 27):
    for i in range(K + 1):
        # Try using k occurrences of the j-th letter
        for k in range(min(i, C[j-1]) + 1):
            dp[i][j] = (dp[i][j] + dp[i-k][j-1] * comb(i, k)) % MOD

# Sum up the results for all lengths from 1 to K
result = 0
for length in range(1, K + 1):
    result = (result + dp[length][26]) % MOD

print(result)