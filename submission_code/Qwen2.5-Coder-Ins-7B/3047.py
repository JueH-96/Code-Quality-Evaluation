class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        from math import isqrt
        from collections import defaultdict
        
        def is_perfect_square(n):
            return isqrt(n) ** 2 == n
        
        def get_prime_factors(n):
            factors = defaultdict(int)
            for i in range(2, isqrt(n) + 1):
                while n % i == 0:
                    factors[i] += 1
                    n //= i
            if n > 1:
                factors[n] += 1
            return factors
        
        def get_product_of_factors(factors):
            product = 1
            for prime, exp in factors.items():
                product *= prime ** (exp // 2)
            return product
        
        n = len(nums)
        dp = [0] * (1 << n)
        for i in range(n):
            dp[1 << i] = nums[i]
        
        for mask in range(1, 1 << n):
            if dp[mask] == 0:
                continue
            prime_factors = defaultdict(int)
            for i in range(n):
                if mask & (1 << i):
                    prime_factors.update(get_prime_factors(nums[i]))
            product_of_factors = get_product_of_factors(prime_factors)
            if is_perfect_square(product_of_factors):
                for j in range(mask):
                    if (mask & j) == 0:
                        dp[mask | j] = max(dp[mask | j], dp[mask] + dp[j])
        
        return max(dp)