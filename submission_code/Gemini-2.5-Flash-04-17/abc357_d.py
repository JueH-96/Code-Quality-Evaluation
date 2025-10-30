# YOUR CODE HERE
# Read N from standard input
N = int(input())

# Modulus
M = 998244353

# Calculate the number of digits of N
L = len(str(N))

# V_N is formed by concatenating the string of N, N times.
# Let S be the string of N, and I be the integer value of S (I = N).
# V_N can be written as:
# V_N = I * 10^(L*(N-1)) + I * 10^(L*(N-2)) + ... + I * 10^L + I * 10^0
# V_N = I * (10^0 + 10^L + 10^(2L) + ... + 10^((N-1)L))
# V_N = N * sum_{k=0}^{N-1} (10^L)^k

# We need to calculate V_N mod M.
# V_N mod M = (N mod M * sum_{k=0}^{N-1} (10^L)^k mod M) mod M

# Let A = 10^L mod M.
# The sum is a geometric series: S_N = sum_{k=0}^{N-1} A^k mod M.

# Calculate A = 10^L mod M
A = pow(10, L, M)

# Calculate the sum S_N = sum_{k=0}^{N-1} A^k mod M
if A == 1:
    # Case where 10^L == 1 (mod M). The sum is 1 + 1 + ... + 1 (N times) mod M.
    S_N_modM = N % M
else:
    # Case where 10^L != 1 (mod M). The sum is (A^N - 1) / (A - 1) mod M.
    # S_N = (A^N - 1) * (A - 1)^-1 (mod M)

    # We need A^N mod M.
    # By properties of modular exponentiation and Fermat's Little Theorem,
    # a^b mod p = a^(b mod p-1) mod p for prime p and a not divisible by p.
    # M is prime. A = 10^L is not divisible by M since M is prime and does not divide 10, and L >= 1.
    # Thus, A^N mod M = A^(N mod M-1) mod M.
    N_exp_for_A = N % (M - 1)

    # Calculate A^N mod M
    AN_modM = pow(A, N_exp_for_A, M)

    # Calculate the numerator: (A^N - 1) mod M
    # Add M before taking modulo to ensure the result is non-negative
    Num = (AN_modM - 1 + M) % M

    # Calculate the denominator: (A - 1) mod M
    Den = (A - 1 + M) % M

    # Calculate the modular multiplicative inverse of Den modulo M.
    # Since M is prime, we use Fermat's Little Theorem: Den^(M-2) mod M
    # Den = 10^L - 1 mod M. For 1 <= L <= 19 (L for 1 <= N <= 10^18),
    # 10^L is not 1 mod M. So Den is guaranteed not to be 0 mod M.
    Den_inv = pow(Den, M - 2, M)

    # Calculate the sum S_N mod M
    S_N_modM = (Num * Den_inv) % M

# The final result is V_N mod M = (N * S_N) mod M
# V_N mod M = (N mod M * S_N mod M) mod M
N_modM = N % M

Result = (N_modM * S_N_modM) % M

# Print the result to standard output
print(Result)