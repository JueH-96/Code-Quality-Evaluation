class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        if n == 0:
            return 0
        
        max_time = sum(time)
        # Initialize DP table with infinity
        dp = [[float('inf')] * (max_time + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 walls painted, 0 cost, 0 time
        
        for i in range(n):
            current_cost = cost[i]
            current_time = time[i]
            # Process the DP table in reverse to avoid overwriting values we still need to process
            for j in range(n, 0, -1):
                for t in range(max_time, -1, -1):
                    if dp[j-1][t] != float('inf'):
                        # Option 1: include this wall in the paid painter's work
                        new_t = t + current_time
                        if new_t <= max_time:
                            if dp[j][new_t] > dp[j-1][t] + current_cost:
                                dp[j][new_t] = dp[j-1][t] + current_cost
        
        min_cost = float('inf')
        # Check all possible states where the paid painter has painted at least one wall
        for j in range(1, n + 1):
            for t in range(max_time + 1):
                if dp[j][t] != float('inf') and t >= (n - j):
                    if dp[j][t] < min_cost:
                        min_cost = dp[j][t]
        
        return min_cost if min_cost != float('inf') else sum(cost)