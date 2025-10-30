import sys

def binom(n, k):
    if k < 0 or k > n:
        return 0
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
A = [int(data[index + i]) for i in range(N)]

MOD = 998244353

# Compute prefix sums P
P = [0] * (N + 1)
for i in range(1, N + 1):
    P[i] = (P[i - 1] + A[i - 1]) % MOD

# Compute prefix sums for each exponent exp from 0 to K
prefix_sums = [None] * (K + 1)
for exp in range(K + 1):
    cum = 0
    prefix = [0] * (N + 1)
    for i in range(N + 1):
        val = pow(P[i], exp, MOD)
        cum = (cum + val) % MOD
        prefix[i] = cum
    prefix_sums[exp] = prefix

# Compute the total sum
total_sum = 0
for m in range(K + 1):
    binom_coeff = binom(K, m)
    sign = (-1) ** (K - m)
    inner_sum = 0
    for j in range(1, N + 1):
        val_j = pow(P[j], m, MOD)
        val_prefix = prefix_sums[K - m][j - 1]
        prod = (val_j * val_prefix) % MOD
        inner_sum = (inner_sum + prod) % MOD
    signed_part = ((binom_coeff * inner_sum) % MOD * sign) % MOD
    signed_part = (signed_part + MOD) % MOD  # Ensure non-negative
    total_sum = (total_sum + signed_part) % MOD

# Output the result
print(total_sum)