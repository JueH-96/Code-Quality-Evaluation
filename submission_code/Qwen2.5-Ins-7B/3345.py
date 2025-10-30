from typing import List
from functools import lru_cache

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dp(index, target):
            if target == 0:
                return 1
            if index == len(nums) or target < 0:
                return 0
            
            # Include nums[index] in the subsequence
            include = dp(index + 1, target - nums[index])
            # Exclude nums[index] from the subsequence
            exclude = dp(index + 1, target)
            
            return (include + exclude) % MOD
        
        total_power = 0
        for i in range(len(nums)):
            # Calculate the power of subsequences starting from nums[i]
            total_power += dp(i, k - nums[i])
            total_power %= MOD
        
        return total_power