class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        max_time_sum = sum(time)
        dp = [[float('inf')] * (max_time_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            current_cost = cost[i-1]
            current_time = time[i-1]
            for j in range(max_time_sum + 1):
                # Option 1: Paint wall i-1 with free painter
                dp[i][j] = min(dp[i][j], dp[i-1][j])
                # Option 2: Paint wall i-1 with paid painter
                if j >= current_time:
                    dp[i][j] = min(dp[i][j], dp[i-1][j - current_time] + current_cost)
                else:
                    dp[i][j] = min(dp[i][j], dp[i-1][0] + current_cost)
                    
        min_total_cost = float('inf')
        for j in range(max_time_sum + 1):
            if dp[n][j] != float('inf'):
                min_total_cost = min(min_total_cost, dp[n][j])
                
        return min_total_cost