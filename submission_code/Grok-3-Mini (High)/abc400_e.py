import sys
import math
import bisect

# Precompute all M with exactly two distinct prime factors up to 10^6
MAX_M = 1000000
omega = [0] * (MAX_M + 1)
for i in range(2, MAX_M + 1):
    if omega[i] == 0:  # i is prime
        for j in range(i, MAX_M + 1, i):
            omega[j] += 1

# Create sorted list of M with exactly two distinct prime factors
candidates = [n for n in range(2, MAX_M + 1) if omega[n] == 2]

# Read all input data
data = list(map(int, sys.stdin.read().split()))

# First element is Q
Q = data[0]

# Process each query
for i in range(Q):
    A = data[i + 1]
    # Compute floor of square root of A
    B = int(math.sqrt(A))
    # Find the index where to insert B+1, which is the first position > B
    idx = bisect.bisect_right(candidates, B)
    # The largest M <= B is at index idx - 1
    M_max = candidates[idx - 1]
    # Compute N = M_max squared
    N = M_max ** 2
    # Output N
    print(N)