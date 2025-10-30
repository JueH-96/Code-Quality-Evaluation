from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[s] will store the maximum length of a subsequence that sums to s
        # Initialize all to -1 (unreachable), except dp[0] = 0 (sum 0 with length 0)
        dp = [-1] * (target + 1)
        dp[0] = 0
        
        for num in nums:
            # Iterate sums in descending order to avoid reuse of the same element
            for s in range(target, num - 1, -1):
                if dp[s - num] != -1:
                    dp[s] = max(dp[s], dp[s - num] + 1)
        
        # If dp[target] is still -1, no subsequence sums to target
        return dp[target] if dp[target] != -1 else -1