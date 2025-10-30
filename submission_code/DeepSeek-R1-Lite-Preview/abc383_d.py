import math
from bisect import bisect_right
import sys

def sieve(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for multiple in range(i*i, limit + 1, i):
                sieve[multiple] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes

def find_max_x(N, power):
    left = 0
    right = N
    while left < right:
        mid = (left + right + 1) // 2
        if mid ** power <= N:
            left = mid
        else:
            right = mid - 1
    return left

def main():
    N = int(sys.stdin.read())
    
    # Count numbers of the form p^8
    n1 = find_max_x(N, 8)
    primes_p8 = sieve(n1)
    count1 = 0
    for p in primes_p8:
        p8 = p ** 8
        if p8 <= N:
            count1 += 1
    
    # Count numbers of the form p^2 * q^2
    n3 = math.isqrt(N)
    primes = sieve(n3)
    n2 = find_max_x(N, 4)
    count2 = 0
    for p in primes:
        if p > n2:
            break
        max_q = n3 // p
        if max_q < p:
            continue
        # Find number of primes q where q > p and q <= max_q
        idx = bisect_right(primes, max_q)
        count_q = idx - bisect_right(primes, p)
        count2 += count_q
    
    total = count1 + count2
    print(total)

if __name__ == "__main__":
    main()