class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Create a dp array to store minimum costs
        # dp[i][j] represents min cost to paint houses 0 to i with house i colored j
        dp = [[float('inf')] * 3 for _ in range(n)]
        
        # Initialize first house costs
        for j in range(3):
            dp[0][j] = cost[0][j]
            
        # Fill dp array for each house
        for i in range(1, n):
            for curr_color in range(3):
                # Get the opposite house index (equidistant from ends)
                opposite = n - 1 - i
                
                # For each possible color of previous house
                for prev_color in range(3):
                    # Skip if current color is same as previous color
                    if curr_color == prev_color:
                        continue
                        
                    # If we're painting a house that has an opposite counterpart
                    if i >= opposite:
                        # Get the color of the opposite house
                        opposite_color = -1
                        for c in range(3):
                            if dp[opposite][c] != float('inf'):
                                opposite_color = c
                                break
                        # Skip if current color would match opposite house color
                        if curr_color == opposite_color:
                            continue
                    
                    # Update minimum cost
                    dp[i][curr_color] = min(
                        dp[i][curr_color],
                        dp[i-1][prev_color] + cost[i][curr_color]
                    )
        
        # Find minimum cost among all possible colors for last house
        min_cost = min(dp[n-1])
        
        # Return -1 if no valid solution found
        return min_cost if min_cost != float('inf') else -1