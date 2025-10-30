from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        end_offers = [[] for _ in range(n)]
        for s, e, g in offers:
            end_offers[e].append((s, g))
        
        dp = [0] * n
        for i in range(n):
            if i > 0:
                dp[i] = dp[i-1]
            for s, gold in end_offers[i]:
                prev_val = dp[s-1] if s - 1 >= 0 else 0
                if prev_val + gold > dp[i]:
                    dp[i] = prev_val + gold
        return dp[n-1]