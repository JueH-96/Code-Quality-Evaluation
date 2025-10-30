import sys

# --- Start of the main logic ---
MOD = 998244353

# Read input
N, K = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Compute prefix sums S_prefix[i] = sum_{k=0}^{i-1} A_k for i=0..N
# S_prefix[0] = 0
S_prefix = [0] * (N + 1)
for i in range(N):
    S_prefix[i + 1] = (S_prefix[i] + A[i]) % MOD

# Compute S_prefix_pow[i][k] = S_prefix[i]^k for i=0..N, k=0..K
S_prefix_pow = [[0] * (K + 1) for _ in range(N + 1)]
for i in range(N + 1):
    S_prefix_pow[i][0] = 1 # Base case: x^0 = 1 (even for x=0). pow(0,0) == 1 in Python.
    for k in range(1, K + 1):
        # Compute iteratively: S_prefix_pow[i][k] = S_prefix[i]^(k-1) * S_prefix[i]
        S_prefix_pow[i][k] = (S_prefix_pow[i][k - 1] * S_prefix[i]) % MOD

# Compute P_k[m][k'] = sum_{i=0}^m S_prefix[i]^k' for m=0..N, k'=0..K
P_k = [[0] * (K + 1) for _ in range(N + 1)]
# P_k[m][k_] represents sum_{i=0..m} S_prefix[i]^{k_}
# m=0: P_k[0][k_] = S_prefix[0]^{k_} = S_prefix_pow[0][k_]
for k_ in range(K + 1):
    P_k[0][k_] = S_prefix_pow[0][k_]

# m=1..N: P_k[m][k_] = P_k[m-1][k_] + S_prefix[m]^{k_} = P_k[m-1][k_] + S_prefix_pow[m][k_]
for m in range(1, N + 1):
    for k_ in range(K + 1):
        P_k[m][k_] = (P_k[m - 1][k_] + S_prefix_pow[m][k_]) % MOD

# Compute binomial coefficients C(K, j) for j=0..K
C = [0] * (K + 1)
C[0] = 1
# Need modular inverse for 1, 2, ..., K
# Using iterative method for inverses: inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
# Requires i > 1
inv = [0] * (K + 1)
if K >= 1:
    # inv[0] is not needed
    inv[1] = 1
    for i in range(2, K + 1):
         inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD

for j in range(1, K + 1):
    C[j] = (C[j - 1] * (K - j + 1) * inv[j]) % MOD

# Compute the final sum
TotalSum = 0
# Formula: sum_{j=0..K} C(K, j) * (-1)^{K-j} * sum_{r=0..N-1} S_prefix[r+1]^j * P_{K-j}(r)
# P_{K-j}(r) = sum_{i=0..r} S_prefix[i]^{K-j} which is P_k[r][K-j]

for j in range(K + 1):
    binom_coeff = C[j]
    sign = 1 if (K - j) % 2 == 0 else -1
    term_j_coeff = (binom_coeff * sign) % MOD
    # Adjust negative result to be in [0, MOD-1)
    term_j_coeff = (term_j_coeff + MOD) % MOD

    sum_over_r = 0
    # r goes from 0 to N-1
    # S_prefix[r+1] corresponds to S_prefix_pow[r+1]
    # P_{K-j}(r) corresponds to P_k[r][K-j]
    for r in range(N):
        S_r_plus_1_pow_j = S_prefix_pow[r + 1][j]
        P_K_minus_j_r = P_k[r][K - j]
        sum_over_r = (sum_over_r + S_r_plus_1_pow_j * P_K_minus_j_r) % MOD

    TotalSum = (TotalSum + term_j_coeff * sum_over_r) % MOD
    # Adjust negative result from adding term_j_coeff
    TotalSum = (TotalSum + MOD) % MOD

# Print the result
print(TotalSum)

# --- End of the main logic ---