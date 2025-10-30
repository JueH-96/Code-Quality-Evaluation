from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        
        for i in range(n):
            if nums[i] == 1:
                dp[i + 1] = dp[i]
            else:
                if i >= 2:
                    dp[i + 1] = dp[i - 2] + 1
                else:
                    return -1
        
        return dp[n]