from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        prefix_sum = [0] * (n + 1)
        stack = [0]
        
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
            dp[i] = dp[i - 1] + 1
            while stack and prefix_sum[i] - prefix_sum[stack[-1]] >= prefix_sum[stack[-1]]:
                j = stack.pop()
                dp[i] = max(dp[i], dp[j] + 1)
            stack.append(i)
        
        return dp[n]