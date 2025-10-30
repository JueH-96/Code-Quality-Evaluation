from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = [float('-inf')] * 5
        dp[0] = 0  # Base case: score with 0 elements selected
        
        for j in range(n):
            for k in range(4, 0, -1):
                if dp[k-1] != float('-inf'):
                    dp[k] = max(dp[k], dp[k-1] + a[k-1] * b[j])
        
        return dp[4]