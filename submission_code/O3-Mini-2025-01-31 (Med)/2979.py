from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Create a list to hold offers ending at each house index.
        offers_by_end = [[] for _ in range(n)]
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))
            
        # dp[i] represents maximum gold up to house i-1.
        dp = [0] * (n + 1)
        
        # Process houses from 0 to n-1, updating the dp array.
        for i in range(n):
            # Option to not sell house i (carry over the best result).
            dp[i + 1] = max(dp[i + 1], dp[i])
            # Process each offer ending at house i
            for start, gold in offers_by_end[i]:
                dp[i + 1] = max(dp[i + 1], dp[start] + gold)
                
        return dp[n]