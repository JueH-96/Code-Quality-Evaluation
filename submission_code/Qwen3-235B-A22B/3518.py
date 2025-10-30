from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [float('-inf')] * 4
        
        for num in b:
            for j in reversed(range(4)):
                if j == 0:
                    current = a[0] * num
                    dp[j] = max(dp[j], current)
                else:
                    if dp[j-1] != float('-inf'):
                        current = dp[j-1] + a[j] * num
                        dp[j] = max(dp[j], current)
        
        return dp[3]