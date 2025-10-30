from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = nums.length
        dp = [[0] * (k + 1) for _ in range(n)]

        # Initialize the first element
        for j in range(k + 1):
            dp[0][j] = 1

        # Fill the DP table
        for i in range(1, n):
            for j in range(k + 1):
                # Not including nums[i]
                dp[i][j] = dp[i-1][j]
                # Including nums[i]
                if nums[i] == nums[i-1]:
                    dp[i][j] = max(dp[i][j], dp[i-1][j] + 1)
                elif j > 0:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)

        # The result is the maximum value in the last row of the DP table
        return max(dp[n-1])