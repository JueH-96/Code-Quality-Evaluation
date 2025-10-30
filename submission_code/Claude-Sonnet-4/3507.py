class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # Find the range of primes we need to check
        # We need primes p such that l <= p² <= r
        # So sqrt(l) <= p <= sqrt(r)
        min_prime = int(math.sqrt(l))
        max_prime = int(math.sqrt(r))
        
        # Adjust min_prime if needed
        if min_prime * min_prime < l:
            min_prime += 1
            
        # Generate primes up to max_prime using Sieve of Eratosthenes
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
        
        primes = sieve(max_prime)
        
        # Count special numbers (squares of primes) in range [l, r]
        special_count = 0
        for prime in primes:
            square = prime * prime
            if l <= square <= r:
                special_count += 1
        
        # Return count of non-special numbers
        total_numbers = r - l + 1
        return total_numbers - special_count