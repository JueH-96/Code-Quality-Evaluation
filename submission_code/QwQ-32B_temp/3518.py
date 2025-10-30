from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        INF = float('-inf')
        dp = [INF] * 5
        dp[0] = 0  # Starting with 0 elements selected, score is 0
        
        for num in b:
            new_dp = list(dp)
            for k in range(1, 5):
                if dp[k - 1] != INF:
                    current_val = dp[k - 1] + a[k - 1] * num
                    if current_val > new_dp[k]:
                        new_dp[k] = current_val
            dp = new_dp
        
        return dp[4]