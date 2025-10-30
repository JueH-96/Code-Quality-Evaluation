import math
import sys

# Read the input N as an integer
N = int(sys.stdin.readline())

# Initialize the count of good integers
count = 0

# A positive integer X is called a good integer if X = 2^a * b^2
# for positive integers a, b (a >= 1, b >= 1).
# Let the prime factorization of b be b = 2^k * m, where m is odd and k >= 0.
# Then b^2 = 2^(2k) * m^2.
# X = 2^a * 2^(2k) * m^2 = 2^(a+2k) * m^2.
# Let alpha = a+2k. Since a >= 1 and k >= 0 (as b >= 1), alpha = a+2k >= 1.
# Since m is the odd part of b and b >= 1, m is an odd integer >= 1.
# Thus, a good integer X must be of the form X = 2^alpha * m^2,
# where alpha >= 1 is an integer and m >= 1 is an odd integer.

# Conversely, if X = 2^alpha * m^2 where alpha >= 1 and m >= 1 is odd.
# We can choose a = alpha and b = m.
# Since alpha >= 1, a >= 1. Since m >= 1 and m is odd, m is a positive integer, so b >= 1.
# Thus, X is a good integer by the definition.

# So, we need to count integers X such that 1 <= X <= N and X = 2^alpha * m^2
# where alpha >= 1 is an integer and m >= 1 is an odd integer.

# We can iterate through possible values of alpha, starting from 1.
# Let P = 2^alpha. We need P * m^2 <= N for some odd m >= 1.
# This implies P must be <= N.

# Start with alpha = 1, P = 2^1 = 2.
P = 2

# Loop through powers of 2, P = 2^alpha, as long as P <= N.
# For each P, we count the number of valid odd m >= 1.
while P <= N:
    # The condition is P * m^2 <= N, which is m^2 <= N / P.
    # We need to count odd integers m >= 1 such that m^2 <= floor(N / P).
    # Let M = floor(N / P) using integer division.
    M = N // P

    # We need to count odd m >= 1 such that m^2 <= M.
    # This means 1 <= m <= floor(sqrt(M)) and m is odd.
    # Let V = floor(sqrt(M)). We use math.isqrt for integer square root.
    # Since P <= N and P >= 2, M = N // P >= 1. So V = math.isqrt(M) >= 1.
    V = math.isqrt(M)

    # The number of odd integers in the range [1, V] is (V + 1) // 2.
    # For example:
    # V=1: {1}, count=1. (1+1)//2=1.
    # V=2: {1}, count=1. (2+1)//2=1.
    # V=3: {1, 3}, count=2. (3+1)//2=2.
    # V=4: {1, 3}, count=2. (4+1)//2=2.
    # V=5: {1, 3, 5}, count=3. (5+1)//2=3.
    num_odd_m = (V + 1) // 2

    # Add the number of good integers found for the current power of 2
    count += num_odd_m

    # Move to the next power of 2: P = 2^(alpha+1).
    # P grows as a Python arbitrary precision integer.
    # The loop condition `P <= N` will handle termination correctly
    # when P exceeds N.
    P *= 2

# Print the final count
print(count)