# YOUR CODE HERE
import sys
import math
from bisect import bisect_right

def main():
    import sys
    import threading
    def sieve(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        return primes

    def solve():
        N = int(sys.stdin.readline())
        N_sqrt = int(N**0.5)+1
        primes = sieve(N_sqrt)

        # Count primes p such that p^8 ≤ N
        limit_p8 = int(N**(1/8)) + 1
        primes_p8 = [p for p in primes if p <= limit_p8 and p**8 <= N]
        count_p8 = len(primes_p8)

        # Count numbers of the form (p*q)^2 where p<q are primes and (p*q)^2 ≤ N
        count_semi_square = 0

        total_primes = len(primes)
        for idx_p, p in enumerate(primes):
            max_q = N_sqrt // p
            # Find index of max_q in primes
            idx_q = bisect_right(primes, max_q)
            # Since q > p, we need to adjust indices
            idx_q = max(idx_q, idx_p+1)
            count = idx_q - (idx_p+1)
            count_semi_square += count

        total_count = count_p8 + count_semi_square
        print(total_count)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()