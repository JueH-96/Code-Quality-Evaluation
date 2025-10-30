from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Initialize a DP table
        dp = [[float('inf')] * 3 for _ in range(n)]
        
        # Base case: fill the first house costs
        for j in range(3):
            dp[0][j] = cost[0][j]
        
        # Fill the DP table
        for i in range(1, n):
            for j in range(3):
                # Ensure no two adjacent houses are painted the same color
                for k in range(3):
                    if j != k:
                        dp[i][j] = min(dp[i][j], dp[i-1][k] + cost[i][j])
        
        # Handle the equidistant condition
        # We need to consider two cases: 
        # 1. The first house is painted with color 1
        # 2. The first house is painted with color 2
        # We will calculate the minimum cost for both cases and return the minimum
        
        min_cost = float('inf')
        
        for first_color in range(3):
            for last_color in range(3):
                if first_color != last_color:
                    # Calculate the cost for the first half
                    first_half_cost = dp[n//2 - 1][first_color]
                    # Calculate the cost for the second half
                    second_half_cost = dp[n - 1][last_color]
                    # Ensure the equidistant condition for the middle houses
                    if (n // 2) % 2 == 0:  # even index
                        if first_color != last_color:
                            min_cost = min(min_cost, first_half_cost + second_half_cost)
        
        return min_cost