import math
import bisect # To efficiently find primes <= M

# Maximum possible value for floor(sqrt(A)) is 10^6.
# We need primes up to 10^6.
MAX_B_PRECOMP = 1000000

# Precompute primes using Sieve of Eratosthenes
IS_PRIME = [True] * (MAX_B_PRECOMP + 1)
IS_PRIME[0] = IS_PRIME[1] = False
for i in range(2, int(math.sqrt(MAX_B_PRECOMP)) + 1):
    if IS_PRIME[i]:
        for j in range(i * i, MAX_B_PRECOMP + 1, i):
            IS_PRIME[j] = False

# Store primes in a list
PRIMES = []
for i in range(2, MAX_B_PRECOMP + 1):
    if IS_PRIME[i]:
        PRIMES.append(i)


def solve():
    A = int(input())
    B = int(math.sqrt(A)) # B <= MAX_B_PRECOMP

    max_X = 0

    # Iterate through one prime factor p and its exponent a
    # p must be a prime <= B
    # We iterate through PRIMES list. p will be <= B due to the check inside.
    for p in PRIMES:
        # Optimization: If current prime p > B, then p^a > B for any a >= 1
        # Since PRIMES is sorted, we can break early.
        if p > B: break

        P = 1 # Represents p^a, initialized to p^0 = 1
        a = 0
        while True:
            a += 1
            
            # Calculate the next power P_next = p^a from P_current = p^(a-1)
            # Check potential overflow before P_next = P * p
            # If a > 1, P holds p^(a-1). Check if P * p > B.
            if a > 1: # Check only for a > 1, as for a=1, P becomes p, handled by p <= B.
                # If P > B // p, then P * p > B
                if P > B // p: 
                     break # p^a would exceed B

            # Calculate P = p^a
            if a == 1:
                 P = p
            else: # a > 1, P holds p^(a-1)
                 P *= p

            # Now P is p^a
            M = B // P
            if M < 2: # Need q >= 2, b >= 1, so q^b >= 2. Need B/P >= 2.
                break

            # Find largest Y = q'^b' <= M where q' is prime, q' != p, b' >= 1
            max_Y_for_P = 0
            
            # Iterate through the base prime q_prime for Y
            # q_prime must be <= M.
            # Find the index of the largest prime <= M in the PRIMES list using binary search.
            # bisect.bisect_right(a, x) returns an insertion point which comes after (to the right of) any existing entries of x in a.
            # So, bisect.bisect_right(PRIMES, M) gives the index of the first prime > M.
            # We need to iterate through primes up to this index.
            q_prime_limit_index = bisect.bisect_right(PRIMES, M)

            for i in range(q_prime_limit_index):
                q_prime = PRIMES[i]

                if q_prime == p: continue # Skip if q' is the same prime as p

                Q_prime = q_prime # Q_prime = q'^b', starting with b'=1
                b_prime = 1
                while True:
                    # Q_prime is q'^b'
                    # Check if current power is too large (Q_prime > M)
                    if Q_prime > M: break 

                    # Q_prime is a candidate for Y = q'^b'
                    max_Y_for_P = max(max_Y_for_P, Q_prime)

                    # Calculate next power of q_prime: Q_prime * q_prime
                    # Check for potential overflow before multiplication Q_prime * q_prime
                    # Need next Q_prime <= M. If Q_prime > M // q_prime, then Q_prime * q_prime > M.
                    if M // q_prime < Q_prime: 
                        break # Q_prime * q_prime would exceed M

                    Q_prime *= q_prime
                    b_prime += 1

            # Finished finding max_Y_for_P for the current P = p^a
            # We need X = p^a * q^b. max_Y_for_P is the largest q^b <= M (where q != p).
            # If max_Y_for_P > 0, we found a valid Y = q^b.
            if max_Y_for_P > 0:
                current_X = P * max_Y_for_P
                # current_X = p^a * q^b. P <= B/M, max_Y_for_P <= M.
                # current_X <= (B/M) * M = B.
                # Python's int handles large numbers.
                max_X = max(max_X, current_X)

            # Loop continues to calculate P = p^(a+1)

    # The largest X <= B with two distinct prime factors is max_X.
    # The answer is the largest 400 number <= A, which is max_X^2.
    print(max_X * max_X)

# Read the number of queries
Q = int(input())

# Process each query
for _ in range(Q):
    solve()