class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # dp[i] = minimum cost to paint at least i walls
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for c, t in zip(cost, time):
            # Consider using paid painter for this wall
            # Iterate in reverse to avoid using the same wall multiple times
            for i in range(n, 0, -1):
                # If we use paid painter for this wall:
                # - We paint 1 + t walls total (1 by paid, t by free)
                # - To paint at least i walls, we need to have painted at least max(0, i - 1 - t) walls before
                prev_walls_needed = max(0, i - 1 - t)
                dp[i] = min(dp[i], dp[prev_walls_needed] + c)
        
        return dp[n]