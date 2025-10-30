# YOUR CODE HERE
MOD = 998244353

def power(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

# Read input
N, M = map(int, input().split())

# Calculate (M-1)^N mod MOD
m_minus_1_pow_n = power(M - 1, N, MOD)

# Calculate the final answer
# Answer = (M-1)^N + (-1)^N * (M-1)
if N % 2 == 0:
    answer = (m_minus_1_pow_n + (M - 1)) % MOD
else:
    answer = (m_minus_1_pow_n - (M - 1) + MOD) % MOD

print(answer)