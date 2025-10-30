class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # Find the smallest integer p such that p^2 >= l
        low = math.isqrt(l)
        if low * low < l:
            low += 1
        
        # Find the largest integer p such that p^2 <= r
        high = math.isqrt(r)
        
        # If there are no p in [low, high], then no special numbers
        if low > high:
            special = 0
        else:
            # Sieve primes up to 'high'
            n = high
            sieve = [True] * (n + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(n**0.5) + 1):
                if sieve[i]:
                    for j in range(i * i, n + 1, i):
                        sieve[j] = False
            
            # Count primes p in [low, high]
            special = sum(1 for p in range(low, high + 1) if sieve[p])
        
        # Total numbers minus the special ones
        return (r - l + 1) - special