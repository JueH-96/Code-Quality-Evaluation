class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # Initialize a DP array with infinity
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 cost to paint 0 walls
        
        for i in range(n):
            current_cost = cost[i]
            current_time = time[i]
            # Iterate from the end to the beginning to avoid overwriting
            for j in range(n, -1, -1):
                # Calculate the new number of walls that can be painted
                new_walls = min(j + current_time + 1, n)
                # Update the DP array
                if dp[j] != float('inf'):
                    dp[new_walls] = min(dp[new_walls], dp[j] + current_cost)
        
        # The answer is the minimum cost to paint at least n walls
        return dp[n]