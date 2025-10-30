import sys
import random

# Use SystemRandom for generating random numbers suitable for hashing
# This provides a more secure source of randomness than the default PRNG,
# reducing the chance of predictable collisions, although for competitive
# programming default random is often sufficient if seeded properly or
# using large numbers.
rng = random.SystemRandom()

# Read N and Q
N, Q = map(int, sys.stdin.readline().split())

# Read arrays A and B (using 0-based indexing internally)
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Generate random 64-bit hashes for values 1 to N.
# We need a unique random hash for each possible value that can appear in A or B.
# Since 1 <= A_i, B_i <= N, we need hashes for values from 1 to N.
# Using two independent hash functions (hashes1 and hashes2) reduces the
# probability of collision for different multisets.
# Store hashes in lists where index v corresponds to value v. Index 0 is unused.
hashes1 = [0] * (N + 1)
hashes2 = [0] * (N + 1)
for v in range(1, N + 1):
    # Generate a random 64-bit integer. Python integers have arbitrary precision,
    # so these are actual random values within the 64-bit range, not modulo 2^64.
    hashes1[v] = rng.getrandbits(64)
    hashes2[v] = rng.getrandbits(64)

# Compute prefix sums of hashes for arrays A and B.
# Prefix sum array P[i] stores the sum of hashes for elements from index 0 up to i-1
# of the original array. This allows calculating the sum of hashes for a range [l, r]
# (0-based) as P[r+1] - P[l].
# We use 1-based indexing for our prefix sum arrays PHA1, PHA2, PHB1, PHB2
# to align with the 1-based query indices l, r, L, R.
# PHA1[i] = sum of hashes1 for elements A[0] through A[i-1].
# PHA1[0] = 0.
PHA1 = [0] * (N + 1)
PHA2 = [0] * (N + 1)
PHB1 = [0] * (N + 1)
PHB2 = [0] * (N + 1)

for i in range(N):
    # A[i] is the element at 0-based index i. In 1-based prefix sums,
    # this element contributes to PHA[i+1].
    PHA1[i + 1] = PHA1[i] + hashes1[A[i]]
    PHA2[i + 1] = PHA2[i] + hashes2[A[i]]
    PHB1[i + 1] = PHB1[i] + hashes1[B[i]]
    PHB2[i + 1] = PHB2[i] + hashes2[B[i]]

# Process Q queries
for _ in range(Q):
    # Read 1-based query indices for ranges in A and B
    l, r, L, R = map(int, sys.stdin.readline().split())

    # The condition for two subsequences to be rearrangements of each other
    # is that they have the same length and the same count of each element.
    # Check if lengths are equal first.
    if (r - l + 1) != (R - L + 1):
        print("No")
        continue

    # If lengths are equal, compare the multisets using hash sums.
    # The subsequence A[l..r] (1-based) corresponds to elements at 0-based indices l-1 through r-1.
    # The sum of hashes for this range is PHA[r] - PHA[l-1] using our 1-based prefix sum array.
    sumA1 = PHA1[r] - PHA1[l - 1]
    sumA2 = PHA2[r] - PHA2[l - 1]

    # The subsequence B[L..R] (1-based) corresponds to elements at 0-based indices L-1 through R-1.
    # The sum of hashes for this range is PHB[R] - PHB[L-1] using our 1-based prefix sum array.
    sumB1 = PHB1[R] - PHB1[L - 1]
    sumB2 = PHB2[R] - PHB2[L - 1]

    # If the hash sums for both hash functions are equal, the multisets
    # of the two subsequences are considered equal with very high probability.
    if sumA1 == sumB1 and sumA2 == sumB2:
        print("Yes")
    else:
        print("No")