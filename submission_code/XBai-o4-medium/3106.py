from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = [-float('inf')] * (target + 1)
        dp[0] = 0  # Base case: 0 sum with 0 elements
        
        for num in nums:
            # Iterate from target down to num to avoid reusing the same element
            for j in range(target, num - 1, -1):
                dp[j] = max(dp[j], dp[j - num] + 1)
        
        return dp[target] if dp[target] != -float('inf') else -1