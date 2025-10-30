import math
import bisect

def generate_primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for start in range(2, int(math.isqrt(limit)) + 1):
        if sieve[start]:
            for multiple in range(start*start, limit + 1, start):
                sieve[multiple] = False
    return [num for num, is_prime in enumerate(sieve) if is_prime]

def count_numbers_with Nine_divisors(N):
    primes = generate_primes(2000000)
    sqrt_N = math.isqrt(N)
    count = 0
    
    # Count p^8 <= N
    for p in primes:
        p8 = p ** 8
        if p8 > N:
            break
        count += 1
    
    # Count p^2 * q^2 <= N, which is p * q <= sqrt(N)
    for p in primes:
        if p * p > sqrt_N:
            break
        max_q = sqrt_N // p
        # Find number of q where p < q <= max_q
        idx_p = bisect.bisect_right(primes, p)
        idx_max_q = bisect.bisect_right(primes, max_q)
        count += idx_max_q - idx_p
    return count

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    print(count_numbers_with Nine_divisors(N))

if __name__ == "__main__":
    main()