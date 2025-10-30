from typing import List

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        sum_time = sum(time)
        max_sum_w = sum_time + n
        INF = 10**9 + 5
        dp = [INF] * (max_sum_w + 1)
        dp[0] = 0
        
        for i in range(n):
            w_i = time[i] + 1
            c_i = cost[i]
            for s in range(max_sum_w, w_i - 1, -1):
                dp[s] = min(dp[s], dp[s - w_i] + c_i)
        
        ans = INF
        for s in range(n, max_sum_w + 1):
            ans = min(ans, dp[s])
        return ans