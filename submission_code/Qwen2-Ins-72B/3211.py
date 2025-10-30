from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        stack = [0]
        
        for i in range(1, n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                dp[i] = dp[stack[-1]] + 1
            stack.append(i)
            
        stack = [n - 1]
        for i in range(n - 2, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                dp[i] = max(dp[i], dp[stack[-1]] + 1)
            stack.append(i)
            
        return max(dp)