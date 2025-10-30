class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # The maximum possible integer whose square is ≤ 10^9 is 31622
        max_limit = 31622
        # Create a list to mark prime status
        is_prime = [True] * (max_limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        
        # Sieve of Eratosthenes to find primes up to 31622
        for i in range(2, int(math.isqrt(max_limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, max_limit + 1, i):
                    is_prime[j] = False
        
        # Gather all primes up to 31622
        primes = [i for i in range(2, max_limit + 1) if is_prime[i]]
        
        # Count how many prime squares fall within [l, r]
        special_count = 0
        for p in primes:
            square = p * p
            if square > r:
                break
            if square >= l:
                special_count += 1
        
        # The total numbers in the range is (r - l + 1)
        # Subtract the count of 'special' (prime squares) to get 'non-special'
        return (r - l + 1) - special_count