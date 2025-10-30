from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [float('-inf')] * 4
        for num in b:
            # Update dp for j=3, 2, 1
            for j in range(3, 0, -1):
                if dp[j-1] != float('-inf'):
                    candidate = dp[j-1] + a[j] * num
                    if candidate > dp[j]:
                        dp[j] = candidate
            # Update dp[0]
            candidate0 = a[0] * num
            if candidate0 > dp[0]:
                dp[0] = candidate0
        return dp[3]