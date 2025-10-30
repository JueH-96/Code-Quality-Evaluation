from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total_power = 0
        prefix_sum = 0
        
        for i in range(n):
            total_power = (total_power + nums[i]**2 * (prefix_sum + nums[i])) % MOD
            prefix_sum = (2 * prefix_sum + nums[i]) % MOD
        
        return total_power