class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Create a 2D array to store the minimum cost of painting each house with a specific color
        dp = [[float('inf')] * 3 for _ in range(n)]
        
        # Initialize the first row of the DP table
        dp[0][0] = cost[0][0]
        dp[0][1] = cost[0][1]
        dp[0][2] = cost[0][2]
        
        # Fill the DP table
        for i in range(1, n):
            for j in range(3):
                # Choose the minimum cost of painting the current house with color j+1
                dp[i][j] = cost[i][j] + min(dp[i-1][(j+1)%3], dp[i-1][(j+2)%3])
        
        # Find the minimum cost of painting the last house
        return min(dp[n-1])