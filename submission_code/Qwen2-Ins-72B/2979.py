from typing import List

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort()
        dp = {}
        
        def dfs(i):
            if i >= len(offers):
                return 0
            if i in dp:
                return dp[i]
            
            res = 0
            res = max(res, dfs(i + 1))
            j = bisect_left(offers, offers[i][1] + 1, lo=i+1, key=lambda x: x[0])
            res = max(res, offers[i][2] + dfs(j))
            dp[i] = res
            return res
        
        return dfs(0)