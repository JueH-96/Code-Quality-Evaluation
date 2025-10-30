from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[-1] * (target + 1) for _ in range(n + 1)]
        
        def dfs(index: int, current_sum: int) -> int:
            if current_sum == target:
                return 0
            if index == n or current_sum > target:
                return -float('inf')
            if dp[index][current_sum] != -1:
                return dp[index][current_sum]
            
            include = dfs(index + 1, current_sum + nums[index]) + 1
            exclude = dfs(index + 1, current_sum)
            dp[index][current_sum] = max(include, exclude)
            return dp[index][current_sum]
        
        result = dfs(0, 0)
        return result if result != -float('inf') else -1