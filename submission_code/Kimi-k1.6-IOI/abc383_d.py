import math
import bisect

def main():
    N = int(input())
    m = math.isqrt(N)
    
    # Generate primes up to m using sieve of Eratosthenes
    if m < 2:
        primes = []
    else:
        sieve = [True] * (m + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, math.isqrt(m) + 1):
            if sieve[i]:
                sieve[i*i : m+1 : i] = [False] * len(sieve[i*i : m+1 : i])
        primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Case 1: Numbers of the form p^8
    count_case1 = 0
    for p in primes:
        if p ** 8 <= N:
            count_case1 += 1
        else:
            break
    
    # Case 2: Numbers of the form p^2 * q^2 where p < q
    count_case2 = 0
    if primes:
        sqrt_m = math.isqrt(m)
        for p in primes:
            if p > sqrt_m:
                break
            max_q = m // p
            if max_q < p:
                continue
            left = bisect.bisect_right(primes, p)
            right = bisect.bisect_right(primes, max_q)
            count_case2 += max(0, right - left)
    
    total = count_case1 + count_case2
    print(total)

if __name__ == "__main__":
    main()