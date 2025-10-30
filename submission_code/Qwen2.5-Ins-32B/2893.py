from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [[0, 0] for _ in range(n)]
        dp[0][nums[0] % 2] = nums[0]
        
        for i in range(1, n):
            parity = nums[i] % 2
            dp[i][parity] = max(dp[i-1][parity] + nums[i], dp[i-1][1 - parity] + nums[i] - x)
            dp[i][1 - parity] = max(dp[i-1][1 - parity], dp[i-1][parity] - x)
        
        return max(dp[n-1][0], dp[n-1][1])