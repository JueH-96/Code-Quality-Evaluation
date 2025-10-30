import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = int(data[index])
    index += 1

# Modulo constant
MOD = 998244353

# Compute modular inverse of N
inv_N = pow(N, MOD - 2, MOD)

# Compute coefficient (N+1)/N mod MOD
coeff = (N + 1) * inv_N % MOD

# Start with T for x = N
T = A[N]

# Compute T for x from N-1 down to 1
for x in range(N - 1, 0, -1):
    T = (A[x] + (coeff * T % MOD)) % MOD

# Compute expected value E0
E0 = (inv_N * T % MOD) % MOD

# Output the result
print(E0)