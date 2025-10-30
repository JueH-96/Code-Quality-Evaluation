from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # dp[i] will hold the best score for choosing the first j elements of a
        # ending at position i in b
        # Initialize for j = 1 (pick a[0])
        dp = [a[0] * b[i] for i in range(n)]
        
        # Use a large negative as minus infinity
        NEG_INF = -10**30
        
        # For j = 2,3,4 (i.e., a[1], a[2], a[3])
        for j in range(1, 4):
            best_prev = NEG_INF
            new_dp = [NEG_INF] * n
            for i in range(n):
                # best_prev is max(dp[0..i-1])
                new_dp[i] = best_prev + a[j] * b[i]
                # update best_prev to include dp[i]
                if dp[i] > best_prev:
                    best_prev = dp[i]
            dp = new_dp
        
        # The answer is the best score after picking all 4 elements
        return max(dp)