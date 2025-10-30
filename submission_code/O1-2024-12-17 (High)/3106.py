from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[s] will store the maximum length of a subsequence that sums to s
        dp = [-1] * (target + 1)
        dp[0] = 0  # Base case: sum 0 can be achieved by an empty subsequence of length 0
        
        for x in nums:
            # Update dp in descending order to avoid using the same element multiple times
            for s in range(target, x - 1, -1):
                if dp[s - x] != -1:
                    dp[s] = max(dp[s], dp[s - x] + 1)
        
        return dp[target] if dp[target] != -1 else -1