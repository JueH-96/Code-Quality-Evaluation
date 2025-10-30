from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = nums.length
        if n == 0:
            return 0

        # Initialize dp and diff arrays
        dp = [1] * n
        diff = [float('inf')] * n

        max_length = 1

        for i in range(1, n):
            for j in range(i):
                current_diff = abs(nums[i] - nums[j])
                if current_diff <= diff[j]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        diff[i] = current_diff
                    elif dp[j] + 1 == dp[i]:
                        diff[i] = min(diff[i], current_diff)
            max_length = max(max_length, dp[i])

        return max_length