import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # Compute a and b
        sqrt_l = math.isqrt(l)
        if sqrt_l * sqrt_l == l:
            a = sqrt_l
        else:
            a = sqrt_l + 1
        b = math.isqrt(r)
        
        if a > b:
            return r - l + 1
        
        # Generate sieve up to b
        is_prime = [True] * (b + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(b)) + 1):
            if is_prime[i]:
                for j in range(i * i, b + 1, i):
                    is_prime[j] = False
        
        # Count primes between a and b inclusive
        count = 0
        for num in range(a, b + 1):
            if is_prime[num]:
                count += 1
        
        return (r - l + 1) - count