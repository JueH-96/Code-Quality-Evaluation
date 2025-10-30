import sys

# Read input from standard input
# Using sys.stdin.readline is generally faster than input() for large inputs
N, M = map(int, sys.stdin.readline().split())

# Define the modulus
MOD = 998244353

# The problem asks for the number of ways to assign integers (colors)
# from {0, 1, ..., M-1} to N people in a circle such that adjacent people
# have different integers. This is equivalent to counting the number of
# proper vertex colorings of a cycle graph with N vertices using M colors.

# The number of proper colorings of a cycle graph with N vertices using M colors
# (the chromatic polynomial of a cycle graph C_N(M)) is given by the formula:
# C_N(M) = (M-1)^N + (-1)^N * (M-1)

# Calculate the base for the power: M-1
# Since the constraints are 2 <= N, M <= 10^6, M is always >= 2, so M-1 >= 1.
base = M - 1

# Calculate the first term: (M-1)^N mod MOD
# We use Python's built-in pow(base, exp, mod) which computes (base**exp) % mod efficiently
# pow(0, N, MOD) for N >= 1 is 0, so it handles M=1 case correctly if constraints allowed it.
# With M >= 2, base is always >= 1.
term1 = pow(base, N, MOD)

# Calculate the second term: (-1)^N * (M-1) mod MOD
# This term depends on the parity of N.
# If N is even, (-1)^N is 1, the term is (M-1).
# If N is odd, (-1)^N is -1, the term is -(M-1).
term2 = base # This represents (M-1)

# The total number of ways is the sum of term1 and the signed term2, modulo MOD.
# The formula is (term1 + (-1)^N * term2) % MOD
if N % 2 == 0:
    # N is even, so we add term2
    result = (term1 + term2) % MOD
else:
    # N is odd, so we subtract term2
    # We need to ensure the result is non-negative when using modulo after subtraction.
    # The formula (a - b) % m is equivalent to (a - b + m) % m for positive m.
    result = (term1 - term2 + MOD) % MOD

# Print the final answer
print(result)