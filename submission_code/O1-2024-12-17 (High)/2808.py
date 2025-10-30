class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        import math
        n = len(cost)
        # dp[i][c] = minimum cost using first i walls (0..i-1)
        #            to achieve coverage c
        # coverage c means we can paint c walls in total
        # coverage = (#walls paid) + sum_of_time_of_paid_walls
        # We only need to track coverage up to n (once c >= n, we can stop).
        
        dp = [[math.inf] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # No walls chosen, coverage = 0, cost = 0
        
        for i in range(1, n + 1):
            c = cost[i - 1]
            t = time[i - 1]
            for cov in range(n + 1):
                # Option 1: do not pay for wall i-1
                dp[i][cov] = min(dp[i][cov], dp[i - 1][cov])
                
                # Option 2: pay for wall i-1
                new_cov = min(n, cov + 1 + t)  # coverage if we pay for wall i-1
                dp[i][new_cov] = min(dp[i][new_cov], dp[i - 1][cov] + c)
        
        return dp[n][n]