class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number is special if it has exactly 2 proper divisors
        # This happens when the number is a square of a prime
        # Because if n = p^2 where p is prime, divisors are: 1, p, p^2
        # So proper divisors are: 1, p (exactly 2)
        
        # Find all primes up to sqrt(r)
        import math
        
        def sieve_of_eratosthenes(limit):
            if limit < 2:
                return []
            
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            
            for i in range(2, int(math.sqrt(limit)) + 1):
                if is_prime[i]:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = False
            
            return [i for i in range(2, limit + 1) if is_prime[i]]
        
        # Get all primes up to sqrt(r)
        max_prime = int(math.sqrt(r))
        primes = sieve_of_eratosthenes(max_prime)
        
        # Count special numbers (squares of primes) in range [l, r]
        special_count = 0
        for prime in primes:
            square = prime * prime
            if l <= square <= r:
                special_count += 1
            elif square > r:
                break
        
        # Total numbers in range
        total_count = r - l + 1
        
        # Return count of non-special numbers
        return total_count - special_count