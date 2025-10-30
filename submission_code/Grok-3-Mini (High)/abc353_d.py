import sys
MOD = 998244353

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [int(data[index + i]) for i in range(N)]

# Compute number of digits for each A[i]
dig = [len(str(a)) for a in A]

# Compute val_mod: 10^dig[i] % MOD for each i
val_mod = [pow(10, d, MOD) for d in dig]

# Compute suffix sum of val_mod
suffix_sum = [0] * (N + 1)
suffix_sum[N] = 0
for k in range(N - 1, -1, -1):
    suffix_sum[k] = (val_mod[k] + suffix_sum[k + 1]) % MOD

# Compute S1: sum over i of A[i] * suffix_sum[i+1] % MOD
S1 = 0
for i in range(N):
    prod = ((A[i] % MOD) * suffix_sum[i + 1]) % MOD
    S1 = (S1 + prod) % MOD

# Compute S2: sum over j of A[j] * j % MOD
S2 = 0
for j in range(N):
    prod2 = ((A[j] % MOD) * j) % MOD
    S2 = (S2 + prod2) % MOD

# Total sum modulo MOD
total_sum = (S1 + S2) % MOD

# Output the result
print(total_sum)