import math
import bisect
import sys

# Generate primes up to 2000000 using Sieve of Eratosthenes
MAX_PRIME_LIMIT = 2000000
is_prime = [True] * (MAX_PRIME_LIMIT + 1)
is_prime[0] = False
is_prime[1] = False
for i in range(2, int(math.sqrt(MAX_PRIME_LIMIT)) + 1):
    if is_prime[i]:
        for j in range(i * i, MAX_PRIME_LIMIT + 1, i):
            is_prime[j] = False
primes = [i for i in range(2, MAX_PRIME_LIMIT + 1) if is_prime[i]]

# Read input from stdin
data = sys.stdin.read().strip()
N = int(data)

# Compute floor of square root and eighth root of N
S = int(math.sqrt(N))
A = int(N ** (1.0 / 8))

# Compute the number of primes <= A (for p^8 <= N)
sum1 = bisect.bisect_right(primes, A)

# Compute the number of pairs (p, q) with p < q, both prime, and p*q <= S (for (p*q)^2 <= N)
sum2 = 0
max_p_val = int(math.sqrt(S))
for p in primes:
    if p > max_p_val:
        break  # Primes are sorted, so we can stop early
    max_q_val = S // p  # Integer division
    num_q = max(0, bisect.bisect_right(primes, max_q_val) - bisect.bisect_right(primes, p))
    sum2 += num_q

# Total number is sum1 + sum2
total = sum1 + sum2

# Output the result
print(total)