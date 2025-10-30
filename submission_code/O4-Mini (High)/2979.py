from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Group offers by their ending house index
        ends = [[] for _ in range(n)]
        for start, end, gold in offers:
            ends[end].append((start, gold))
        
        # dp[i] = max gold we can earn by selling houses in range [0..i-1]
        dp = [0] * (n + 1)
        
        # Iterate through each house index i (0-based)
        for i in range(n):
            # Option 1: skip house i, carry forward previous best
            dp[i + 1] = max(dp[i + 1], dp[i])
            # Option 2: sell an offer that ends at house i
            for start, gold in ends[i]:
                dp[i + 1] = max(dp[i + 1], dp[start] + gold)
        
        # dp[n] is the best we can do for houses [0..n-1]
        return dp[n]