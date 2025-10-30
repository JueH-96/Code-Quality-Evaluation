from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # dp[t] holds the maximum length of a subsequence that sums to t.
        # Initialize dp[0] = 0 (empty subsequence) and rest as -1 (unreachable).
        dp = [-1] * (target + 1)
        dp[0] = 0

        # Process each number in nums, ensuring each is used at most once.
        for num in nums:
            # We iterate backwards over possible sums to avoid reusing the current element.
            for t in range(target, num - 1, -1):
                if dp[t - num] != -1:
                    dp[t] = max(dp[t], dp[t - num] + 1)
        
        return dp[target] if dp[target] != -1 else -1