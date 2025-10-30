from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        dp = [1] * n
        
        max_consecutive = 1
        
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                dp[i] = dp[i - 1]
            elif nums[i] == nums[i - 1] + 1:
                dp[i] = dp[i - 1] + 1
            elif nums[i] == nums[i - 1] + 2:
                dp[i] = 2
                if i > 1 and nums[i - 1] == nums[i - 2] + 1:
                    dp[i] = dp[i - 2] + 2
            
            max_consecutive = max(max_consecutive, dp[i])
        
        return max_consecutive