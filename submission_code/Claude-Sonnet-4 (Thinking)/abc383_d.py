import math
import bisect

def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

N = int(input())

# Case 1: p^8 <= N
max_p_case1 = int(N**(1/8)) + 2
count_case1 = 0
if max_p_case1 >= 2:
    primes_case1 = sieve(max_p_case1)
    for p in primes_case1:
        if p**8 <= N:
            count_case1 += 1

# Case 2: p1^2 * p2^2 <= N where p1 < p2
max_prime = int(math.sqrt(N // 4)) + 2
count_case2 = 0
if max_prime >= 2:
    primes = sieve(max_prime)
    for p1 in primes:
        p1_squared = p1 * p1
        if p1_squared * p1_squared > N:
            break
        max_p2 = int(math.sqrt(N // p1_squared))
        # Count primes p2 in the range (p1, max_p2]
        left = bisect.bisect_right(primes, p1)
        right = bisect.bisect_right(primes, max_p2)
        count_case2 += right - left

print(count_case1 + count_case2)