class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # We will use dynamic programming to solve this problem.
        # dp[i][c] will represent the minimum cost to paint houses from 0 to i and house i using color c.
        # We need to consider the constraints for adjacent houses and equidistant houses.
        
        # Initialize dp array
        dp = [[float('inf')] * 3 for _ in range(n)]
        
        # Base case: first house can be painted with any color with its respective cost
        for c in range(3):
            dp[0][c] = cost[0][c]
        
        # Fill the dp table
        for i in range(1, n):
            for c1 in range(3):  # Current house color
                for c0 in range(3):  # Previous house color
                    if c1 != c0:  # No two adjacent houses are painted the same color
                        # Equidistant house index
                        j = n - 1 - i
                        if i <= j:  # Only process if the equidistant house is ahead in the list
                            for c2 in range(3):  # Equidistant house color
                                if c1 != c2:  # Equidistant houses are not the same color
                                    if i == j:  # If it's the same house (happens when n/2 is odd)
                                        dp[i][c1] = min(dp[i][c1], dp[i-1][c0] + cost[i][c1])
                                    else:
                                        dp[i][c1] = min(dp[i][c1], dp[i-1][c0] + cost[i][c1] + cost[j][c2])
                                        dp[j][c2] = min(dp[j][c2], dp[i-1][c0] + cost[i][c1] + cost[j][c2])
        
        # The answer is the minimum cost to paint all houses, considering the last house can be any color
        return min(dp[n-1])