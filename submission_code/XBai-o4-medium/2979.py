from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Preprocess offers by their end points
        ends = [[] for _ in range(n)]
        for start, end, gold in offers:
            ends[end].append((start, gold))
        
        # Initialize the dp array
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Default case: take the previous maximum
            dp[i] = dp[i-1]
            # Check all offers ending at house i-1
            for start, gold in ends[i-1]:
                current = dp[start] + gold
                if current > dp[i]:
                    dp[i] = current
        
        return dp[n]