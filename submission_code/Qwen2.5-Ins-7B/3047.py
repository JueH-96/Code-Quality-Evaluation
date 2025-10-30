from math import isqrt
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(n: int) -> bool:
            return isqrt(n) ** 2 == n
        
        n = len(nums)
        factors = [0] * n
        for i in range(n):
            factors[i] = nums[i]
            for j in range(2, isqrt(nums[i]) + 1):
                if nums[i] % j == 0:
                    factors[i] //= j
                    while nums[i] % j == 0:
                        nums[i] //= j
                    factors[i] *= j
            if nums[i] > 1:
                factors[i] *= nums[i]
        
        dp = [0] * (n + 1)
        for i in range(n):
            for j in range(n, 0, -1):
                if factors[i] % j == 0 and is_perfect_square(factors[i] // j):
                    dp[j] = max(dp[j], dp[j - 1] + nums[i])
        
        return max(dp)