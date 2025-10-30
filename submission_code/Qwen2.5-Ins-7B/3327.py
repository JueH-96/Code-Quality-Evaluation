from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i, x in enumerate(nums) if x == 1]
        m = len(ones)
        if m < k:
            return -1
        
        dp = [[float('inf')] * (maxChanges + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        
        for i in range(1, m + 1):
            for j in range(min(i, maxChanges) + 1):
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + ones[i - 1] - ones[i - 2])
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + ones[i - 1] - ones[i - 2])
        
        res = float('inf')
        for j in range(min(m, maxChanges) + 1):
            res = min(res, dp[m][j] + m - k)
        
        return res