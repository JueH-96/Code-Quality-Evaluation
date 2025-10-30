class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # dp[i][j] represents minimum cost to paint houses 0 to i with house i having color j
        dp = [[float('inf')] * 3 for _ in range(n)]
        
        # Initialize first house
        for j in range(3):
            dp[0][j] = cost[0][j]
        
        # For each house from 1 to n-1
        for i in range(1, n):
            mirror_pos = n - 1 - i  # Position of equidistant house
            
            if i <= n // 2:  # We're in the first half, so we can choose freely (considering adjacent constraint)
                for j in range(3):  # Current house color
                    for prev_j in range(3):  # Previous house color
                        if j != prev_j:  # Adjacent houses can't have same color
                            dp[i][j] = min(dp[i][j], dp[i-1][prev_j] + cost[i][j])
            else:  # We're in the second half, need to consider equidistant constraint
                for j in range(3):  # Current house color
                    for prev_j in range(3):  # Previous house color
                        if j != prev_j:  # Adjacent houses can't have same color
                            # Also check equidistant constraint
                            mirror_color = None
                            # Find what color the equidistant house has
                            for k in range(3):
                                if dp[mirror_pos][k] != float('inf'):
                                    # We need to check if this configuration is valid
                                    # But we need a different approach since we don't know the exact color
                                    pass
                            dp[i][j] = min(dp[i][j], dp[i-1][prev_j] + cost[i][j])
        
        # This approach is getting complex. Let me try a different method.
        # Since houses are paired (i, n-1-i), I'll solve for each pair simultaneously.
        
        # Reset and use a different approach
        # dp[i][c1][c2] = min cost for first i pairs where pair i has colors (c1, c2)
        # where c1 is color of house i, c2 is color of house n-1-i
        
        pairs = n // 2
        dp = [[[float('inf')] * 3 for _ in range(3)] for _ in range(pairs)]
        
        # Initialize first pair (houses 0 and n-1)
        for c1 in range(3):
            for c2 in range(3):
                if c1 != c2:  # Equidistant houses must have different colors
                    dp[0][c1][c2] = cost[0][c1] + cost[n-1][c2]
        
        # Fill for remaining pairs
        for i in range(1, pairs):
            left_house = i
            right_house = n - 1 - i
            
            for c1 in range(3):  # Color of left house in current pair
                for c2 in range(3):  # Color of right house in current pair
                    if c1 != c2:  # Equidistant constraint
                        for prev_c1 in range(3):  # Color of left house in previous pair
                            for prev_c2 in range(3):  # Color of right house in previous pair
                                # Check adjacent constraints
                                if c1 != prev_c1 and c2 != prev_c2:
                                    dp[i][c1][c2] = min(dp[i][c1][c2], 
                                                       dp[i-1][prev_c1][prev_c2] + cost[left_house][c1] + cost[right_house][c2])
        
        # Find minimum cost from last pair
        result = float('inf')
        for c1 in range(3):
            for c2 in range(3):
                result = min(result, dp[pairs-1][c1][c2])
        
        return result