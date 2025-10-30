import sys
data = sys.stdin.read().split()
N = int(data[0])
M = int(data[1])
MOD = 998244353

H_sum = 0
for k in range(1, N + 1):
    inv_k = pow(k, MOD - 2, MOD)
    H_sum = (H_sum + inv_k) % MOD

val = ((2 * N % MOD) * H_sum % MOD - 1 + MOD) % MOD
M_squared = (M * M) % MOD
result = (val * M_squared) % MOD

print(result)