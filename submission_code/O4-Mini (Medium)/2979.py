from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Group offers by their start position
        offers_by_start = [[] for _ in range(n)]
        for s, e, w in offers:
            offers_by_start[s].append((e, w))
        
        # dp[i] = max profit we can get selling houses in the range [i..n-1]
        # dp[n] = 0 (no houses left)
        dp = [0] * (n + 1)
        
        # Fill dp from right to left
        for i in range(n - 1, -1, -1):
            # Option 1: skip house i
            best = dp[i + 1]
            # Option 2: take any offer starting at i
            for end, gold in offers_by_start[i]:
                best = max(best, gold + dp[end + 1])
            dp[i] = best
        
        return dp[0]