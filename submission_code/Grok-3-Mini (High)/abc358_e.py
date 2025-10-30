import sys

# Read input
data = sys.stdin.read().split()
K = int(data[0])
C_list = [int(data[i]) for i in range(1, 27)]

mod = 998244353

# Precompute factorial
fact = [1] * (K + 1)
for i in range(1, K + 1):
    fact[i] = (fact[i - 1] * i) % mod

# Precompute inverse factorial
inv_fact = [0] * (K + 1)
for i in range(K + 1):
    inv_fact[i] = pow(fact[i], mod - 2, mod)

# Start with polynomial [1, 0, 0, ..., 0] of size K+1
current_poly = [0] * (K + 1)
current_poly[0] = 1

# Multiply the polynomial by each P_i(x)
for c in C_list:
    if c == 0:
        continue  # Skip if C_i == 0, as it multiplies by 1
    # Create B coefficients for P_i(x)
    B = [0] * (K + 1)
    for t in range(K + 1):
        if t <= c:
            B[t] = inv_fact[t]
        else:
            B[t] = 0  # B[t] is 0 for t > c
    # Create new polynomial after multiplication
    new_poly = [0] * (K + 1)
    for m in range(K + 1):
        sum_val = 0
        for j in range(m + 1):  # j from 0 to m
            prod = (current_poly[j] * B[m - j]) % mod
            sum_val += prod
            sum_val %= mod
        new_poly[m] = sum_val
    # Update current polynomial
    current_poly = new_poly

# Compute the sum for lengths from 1 to K
ans = 0
for m in range(1, K + 1):
    temp = (fact[m] * current_poly[m]) % mod
    ans += temp
    ans %= mod

# Output the answer
print(ans)