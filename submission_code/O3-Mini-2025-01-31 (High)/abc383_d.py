import sys
import math
import bisect

def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    r = int(n**0.5) + 1
    for i in range(2, r):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    # A positive integer n has exactly 9 divisors if and only if:
    # (1) n = p^8, for some prime p, because (8+1) = 9, or 
    # (2) n = p^2 * q^2, where p and q are distinct primes, because (2+1)*(2+1)=9.
    #
    # For n = p^8 <= N, the maximum possible prime p satisfies p^8 <= N.
    # For n = p^2 * q^2 <= N, note that (p^2 * q^2) <= N  if and only if (p*q)^2 <= N,
    # i.e. p * q <= sqrt(N). And we only consider distinct primes with p < q.
    
    # Case 1: Count numbers of the form p^8.
    count_case1 = 0
    # The candidate primes for p must be at most floor(N^(1/8)).
    limit_p8 = int(N ** (1/8)) + 1
    small_primes = sieve(limit_p8)
    for p in small_primes:
        if p ** 8 <= N:
            count_case1 += 1
        else:
            break

    # Case 2: Count numbers of the form p^2 * q^2.
    # We need p*q <= sqrt(N). Let M = isqrt(N) (largest integer m with m*m <= N).
    M = math.isqrt(N)
    primes_up_to_M = sieve(M)
    
    count_case2 = 0
    # For each prime p (with index i), count primes q (with index > i) with q <= M // p.
    for i, p in enumerate(primes_up_to_M):
        max_q = M // p
        # Using bisect_right to find the index of the first prime greater than max_q.
        j = bisect.bisect_right(primes_up_to_M, max_q)
        # Only count q that come after p to ensure p < q.
        if j > i + 1:
            count_case2 += (j - (i + 1))
    
    result = count_case1 + count_case2
    print(result)

if __name__ == '__main__':
    main()