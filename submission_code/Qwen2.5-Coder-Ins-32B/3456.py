from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n)]
        
        for i in range(n):
            for j in range(k + 1):
                dp[i][j] = 1  # At least the element itself
                if i > 0:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j])
                    if nums[i] == nums[i - 1]:
                        dp[i][j] = max(dp[i][j], dp[i - 1][j] + 1)
                    elif j > 0:
                        dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
        
        return max(dp[n - 1])