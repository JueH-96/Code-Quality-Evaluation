from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                dp[i][j] = dp[i - 1][j]
                if i > 1:
                    for l in range(i - 1):
                        if nums[l - 1] != nums[i - 1]:
                            if j > 0:
                                dp[i][j] = max(dp[i][j], dp[l][j - 1] + 1)
                        else:
                            dp[i][j] = max(dp[i][j], dp[l][j] + 1)
        
        return max(dp[n])