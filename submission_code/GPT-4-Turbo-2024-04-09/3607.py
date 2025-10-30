class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from math import isqrt, gcd
        from functools import lru_cache
        
        # Helper function to find the greatest proper divisor
        @lru_cache(None)
        def greatest_proper_divisor(x):
            if x == 1:
                return -1  # 1 has no proper divisors
            for i in range(2, isqrt(x) + 1):
                if x % i == 0:
                    return x // i
            return 1  # If no divisors found, return 1 (for prime numbers)
        
        n = len(nums)
        operations = 0
        
        # We need to ensure the array becomes non-decreasing
        for i in range(n - 1):
            while nums[i] > nums[i + 1]:
                gpd = greatest_proper_divisor(nums[i])
                if gpd == -1:
                    return -1  # If no proper divisor, we can't proceed
                nums[i] //= gpd
                operations += 1
                if nums[i] <= nums[i + 1]:
                    break
            # If after all possible divisions, nums[i] is still greater, it's impossible
            if nums[i] > nums[i + 1]:
                return -1
        
        return operations