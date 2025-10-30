from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] == nums[j] + 1:
                    dp[i] = max(dp[i], dp[j] + 1)
                elif nums[i] == nums[j]:
                    dp[i] = max(dp[i], dp[j])
        return max(dp)