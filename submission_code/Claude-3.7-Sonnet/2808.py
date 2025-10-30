class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # dp[i][j] = min cost to paint j walls using first i walls
        dp = [[float('inf')] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(n + 1):
                # Option 1: Don't use the paid painter for wall i
                dont_paint_cost = dp[i-1][j]
                
                # Option 2: Use the paid painter for wall i
                # When paid painter paints wall i, we gain capacity to paint 1+time[i-1] walls
                # (1 from the paid painter + time[i-1] from the free painter)
                paint_cost = dp[i-1][max(0, j - (1 + time[i-1]))] + cost[i-1]
                
                dp[i][j] = min(dont_paint_cost, paint_cost)
        
        return dp[n][n]