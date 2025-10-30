from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            c = cost[i]
            t = time[i] + 1
            # Iterate in reverse to prevent reusing the same item
            for s in range(n, -1, -1):
                if dp[s] != INF:
                    new_s = s + t
                    if new_s > n:
                        new_s = n
                    if dp[new_s] > dp[s] + c:
                        dp[new_s] = dp[s] + c
        
        return dp[n]