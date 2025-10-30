class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Initialize a dp array where dp[i][j] represents the minimum cost to paint up to house i with color j
        dp = [[0] * 3 for _ in range(n)]
        
        # Base case: the cost of painting the first house with each color
        for j in range(3):
            dp[0][j] = cost[0][j]
        
        # Fill the dp table
        for i in range(1, n):
            for j in range(3):
                # Calculate the minimum cost for painting house i with color j
                # It cannot be the same as the previous house's color
                dp[i][j] = cost[i][j] + min(dp[i-1][(j+1)%3], dp[i-1][(j+2)%3])
        
        # Now we need to ensure the equidistant condition
        # We will check the minimum cost for the last house with each color
        # and ensure it doesn't violate the equidistant condition with the first house
        min_cost = float('inf')
        for j in range(3):
            for k in range(3):
                if j != k:  # Ensure the first and last house are not the same color
                    min_cost = min(min_cost, dp[n-1][j] + cost[0][k] - cost[0][j])
        
        return min_cost