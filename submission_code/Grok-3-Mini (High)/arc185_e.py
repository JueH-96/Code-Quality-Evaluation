import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A_vals = list(map(int, data[index:index + N]))

MOD = 998244353
MAX_A = 100005  # Sufficiently large to cover all A_i

# Precompute Euler's totient function phi
phi = list(range(MAX_A))
for i in range(2, MAX_A):
    if phi[i] == i:  # i is prime
        for j in range(i, MAX_A, i):
            phi[j] -= phi[j] // i

# Precompute divisors for each number up to MAX_A-1
divisors_list = [[] for _ in range(MAX_A)]
for i in range(1, MAX_A):
    for j in range(i, MAX_A, i):
        divisors_list[j].append(i)  # i is a divisor of j

# Precompute powers of 2 modulo MOD
pow2 = [0] * N
if N > 0:
    pow2[0] = 1
    for i in range(1, N):
        pow2[i] = (pow2[i - 1] * 2) % MOD

# Initialize cumulative sum for each divisor
cum_d = [0] * MAX_A

# Initialize T
T = 0

# Add elements one by one and compute T for each m
for pos in range(1, N + 1):  # pos from 1 to N
    val = A_vals[pos - 1]  # Get A_pos
    # Get divisors of val
    divs = divisors_list[val]
    # Compute S_add
    S_add = 0
    for d in divs:
        prod = (phi[d] * cum_d[d] % MOD)
        S_add = (S_add + prod) % MOD
    # Update T
    T = (2 * T + S_add) % MOD
    # Output T for current m
    print(T)
    # Update cum_d for each divisor d of val
    add_val = pow2[pos - 1]  # 2^{pos-1} mod MOD
    for d in divs:
        cum_d[d] = (cum_d[d] + add_val) % MOD