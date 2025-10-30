from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        mono_stack = []
        
        for i in range(n):
            while mono_stack and nums[mono_stack[-1]] <= nums[i]:
                j = mono_stack.pop()
                dp[i] = max(dp[i], dp[j] + (i - j) * nums[j])
            if mono_stack:
                dp[i] = max(dp[i], (i - mono_stack[-1]) * nums[mono_stack[-1]])
            mono_stack.append(i)
        
        return dp[-1]