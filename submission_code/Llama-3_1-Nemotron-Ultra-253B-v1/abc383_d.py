import bisect
import math

def sieve(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes

def main():
    N = int(input().strip())
    sqrt_N = math.isqrt(N)
    primes = sieve(sqrt_N)
    
    count1 = 0
    for p in primes:
        if p ** 8 > N:
            break
        count1 += 1
    
    max_p = math.isqrt(sqrt_N)
    count2 = 0
    for p in primes:
        if p > max_p:
            break
        m = sqrt_N // p
        idx_p = bisect.bisect_right(primes, p)
        idx_m = bisect.bisect_right(primes, m)
        count2 += max(0, idx_m - idx_p)
    
    print(count1 + count2)

if __name__ == "__main__":
    main()