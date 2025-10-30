import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        limit = 31623
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[p]:
                for i in range(p * p, limit + 1, p):
                    is_prime[i] = False
        
        def isqrt(n):
            if n < 0:
                raise ValueError("square root not defined for negative numbers")
            if n == 0:
                return 0
            x = int(math.sqrt(n))
            y = (x + n // x) // 2
            while y < x:
                x = y
                y = (x + n // x) // 2
            return x
            
        start_prime_index = isqrt(l)
        if start_prime_index * start_prime_index < l:
            start_prime_index += 1
        end_prime_index = isqrt(r)
        
        special_count = 0
        for p in range(start_prime_index, end_prime_index + 1):
            if is_prime[p]:
                special_count += 1
                
        return (r - l + 1) - special_count