class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number is "special" if and only if it is the square of a prime.
        # Reason:
        #   A prime square p^2 has exactly 3 divisors: 1, p, and p^2.
        #   Thus, it has exactly 2 proper divisors (1 and p).
        
        # We want the count of numbers in [l, r] that are NOT special.
        # So we need to count how many squares of primes lie in [l, r],
        # then subtract that from the total numbers in [l, r].
        
        # If r < 4, there are no prime squares in the interval:
        if r < 4:
            return r - l + 1
        
        # We only need to consider primes up to sqrt(r).
        import math
        
        upper = int(math.isqrt(r))  # largest prime to consider
        # Sieve of Eratosthenes to get primes up to upper
        sieve = [True]*(upper+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(upper))+1):
            if sieve[i]:
                for multiple in range(i*i, upper+1, i):
                    sieve[multiple] = False
        primes = [i for i in range(2, upper+1) if sieve[i]]
        
        count_special = 0
        # Count prime squares in [l, r]
        for p in primes:
            psq = p*p
            if psq > r:
                break
            if psq >= l:
                count_special += 1
        
        total_nums = r - l + 1
        return total_nums - count_special