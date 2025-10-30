class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # Special numbers are squares of primes
        # Find range of primes whose squares could be in [l, r]
        min_prime = math.ceil(math.sqrt(l))
        max_prime = int(math.sqrt(r))
        
        if min_prime > max_prime:
            # No special numbers in range
            return r - l + 1
        
        # Count primes in [min_prime, max_prime] using sieve
        special_count = self.count_primes_in_range(min_prime, max_prime)
        
        # Total numbers minus special numbers
        return (r - l + 1) - special_count
    
    def count_primes_in_range(self, start, end):
        if end < 2 or start > end:
            return 0
        
        # Sieve of Eratosthenes
        is_prime = [True] * (end + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(end**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, end + 1, i):
                    is_prime[j] = False
        
        # Count primes in [start, end]
        count = 0
        for i in range(max(2, start), end + 1):
            if is_prime[i]:
                count += 1
        
        return count