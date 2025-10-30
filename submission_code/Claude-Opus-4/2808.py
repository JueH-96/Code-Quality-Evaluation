class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # dp[j] = minimum cost to paint at least j walls
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            # Process each wall
            new_dp = dp[:]
            for j in range(n + 1):
                if dp[j] != float('inf'):
                    # If paid painter paints wall i
                    # We get 1 + time[i] walls painted total
                    walls_painted = min(j + 1 + time[i], n)
                    new_dp[walls_painted] = min(new_dp[walls_painted], dp[j] + cost[i])
            dp = new_dp
        
        return dp[n]