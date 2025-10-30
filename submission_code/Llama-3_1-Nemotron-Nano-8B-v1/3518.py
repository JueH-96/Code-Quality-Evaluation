from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [0] + [float('-inf')] * 4  # dp[0] is 0, others are -inf
        for num in b:
            for k in range(4, 0, -1):
                if dp[k-1] != float('-inf') and dp[k-1] + a[k-1] * num > dp[k]:
                    dp[k] = dp[k-1] + a[k-1] * num
        return dp[4]