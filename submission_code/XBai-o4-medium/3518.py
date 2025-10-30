from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [0] + [-float('inf')] * 4  # dp[0], dp[1], dp[2], dp[3], dp[4]
        for num in b:
            for k in range(4, 0, -1):
                current_val = dp[k-1] + a[k-1] * num
                if current_val > dp[k]:
                    dp[k] = current_val
        return dp[4]