from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # dp[i][d] represents the length of the longest subsequence ending at index i with last difference d
        # Since the differences can be up to 300, we can manage with a dictionary for each index
        dp = [dict() for _ in range(n)]
        max_len = 1
        for i in range(n):
            dp[i][0] = 1  # subsequence with only nums[i]
            for j in range(i):
                diff = abs(nums[i] - nums[j])
                # Iterate through all possible previous differences
                for prev_diff in dp[j]:
                    if prev_diff >= diff:
                        if diff in dp[i]:
                            dp[i][diff] = max(dp[i][diff], dp[j][prev_diff] + 1)
                        else:
                            dp[i][diff] = dp[j][prev_diff] + 1
                        max_len = max(max_len, dp[i][diff])
        return max_len