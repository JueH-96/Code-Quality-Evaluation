from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = [[-float('inf')] * 5 for _ in range(n)]
        for i in range(n):
            dp[i][1] = a[0] * b[i]
        for k in range(2, 5):
            for i in range(k - 1, n):
                for j in range(k - 2, i):
                    if dp[j][k - 1] != -float('inf'):
                        dp[i][k] = max(dp[i][k], dp[j][k - 1] + a[k - 1] * b[i])
        max_score = -float('inf')
        for i in range(3, n):
            max_score = max(max_score, dp[i][4])
        return max_score