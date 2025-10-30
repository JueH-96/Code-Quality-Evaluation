from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Create a list for offers ending at each house index.
        offers_by_end = [[] for _ in range(n)]
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))
        
        # dp[i] will store the maximum gold we can earn considering houses [0, i-1].
        dp = [0] * (n + 1)
        
        # Iterate over each house and update dp based on the offers ending at that house.
        for i in range(n):
            # Option 1: skip selling at house i so dp[i+1] is at least dp[i]
            dp[i+1] = max(dp[i+1], dp[i])
            # Option 2: for each offer ending at house i, try selling those houses.
            for start, gold in offers_by_end[i]:
                dp[i+1] = max(dp[i+1], dp[start] + gold)
                
        return dp[n]