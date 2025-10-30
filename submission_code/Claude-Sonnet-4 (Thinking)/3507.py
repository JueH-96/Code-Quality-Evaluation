class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # Find the range of possible primes
        min_p = math.ceil(math.sqrt(l))
        max_p = int(math.sqrt(r))
        
        if min_p > max_p:
            # No special numbers in range
            return r - l + 1
        
        # Generate primes up to max_p using Sieve of Eratosthenes
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
        
        primes = sieve(max_p)
        
        # Count special numbers (squares of primes) in range [l, r]
        special_count = sum(1 for p in primes if p >= min_p)
        
        return r - l + 1 - special_count