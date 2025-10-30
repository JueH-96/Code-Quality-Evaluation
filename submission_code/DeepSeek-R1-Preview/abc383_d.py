import math
import bisect

def sieve(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes

n = int(input())
primes = sieve(math.isqrt(n))

count_p8 = 0
for p in primes:
    p8 = p ** 8
    if p8 > n:
        break
    count_p8 += 1

count_p2q2 = 0
for i, p in enumerate(primes):
    p_squared = p * p
    if p_squared > n:
        break
    max_q_squared = n // p_squared
    if max_q_squared < 1:
        break
    max_q = math.isqrt(max_q_squared)
    if max_q <= p:
        continue
    insertion_point = bisect.bisect_right(primes, max_q)
    q_max_idx = insertion_point - 1
    start_idx = i + 1
    if q_max_idx >= start_idx:
        count_p2q2 += q_max_idx - start_idx + 1

total = count_p8 + count_p2q2
print(total)