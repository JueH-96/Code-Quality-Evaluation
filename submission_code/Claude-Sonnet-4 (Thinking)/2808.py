class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # dp[j] = minimum cost to "cover" at least j walls
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            # Process in reverse order to avoid using the same wall multiple times
            for j in range(n, 0, -1):
                walls_covered = time[i] + 1  # wall i + time[i] walls by free painter
                prev_needed = max(0, j - walls_covered)
                dp[j] = min(dp[j], dp[prev_needed] + cost[i])
        
        return dp[n]