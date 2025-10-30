import math
import bisect

n = int(input())
if n < 1:
    print(0)
    exit()

sqrt_n = math.isqrt(n)
sieve = [True] * (sqrt_n + 1)
sieve[0] = sieve[1] = False
for i in range(2, int(math.isqrt(sqrt_n)) + 1):
    if sieve[i]:
        sieve[i*i : sqrt_n+1 : i] = [False] * len(sieve[i*i : sqrt_n+1 : i])
primes = [i for i, is_prime in enumerate(sieve) if is_prime]

count_p8 = 0
for p in primes:
    if p ** 8 > n:
        break
    count_p8 += 1

count_p2q2 = 0
for p in primes:
    s = p * p
    if s > n:
        break
    max_val = n // s
    if max_val < 2:
        continue
    q_max = math.isqrt(max_val)
    if q_max <= p:
        continue
    low = bisect.bisect_right(primes, p)
    high = bisect.bisect_right(primes, q_max)
    count_p2q2 += high - low

print(count_p8 + count_p2q2)