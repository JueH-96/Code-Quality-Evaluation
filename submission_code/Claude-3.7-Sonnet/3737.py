class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Initialize dp array
        # dp[i][a][b] represents min cost to paint pairs 0 to i
        # where house i is painted with color a and house n-1-i with color b
        dp = [[[float('inf') for _ in range(3)] for _ in range(3)] for _ in range(n // 2)]
        
        # Base case: cost of painting the first pair (houses 0 and n-1)
        for a in range(3):
            for b in range(3):
                if a != b:  # Equidistant houses must have different colors
                    dp[0][a][b] = cost[0][a] + cost[n-1][b]
        
        # Fill in the dp array for the rest of the pairs
        for i in range(1, n // 2):
            for a in range(3):
                for b in range(3):
                    if a != b:  # Equidistant houses must have different colors
                        for a_prev in range(3):
                            for b_prev in range(3):
                                # Adjacent houses must have different colors
                                # and previous state must be valid
                                if a_prev != a and b_prev != b and dp[i-1][a_prev][b_prev] != float('inf'):
                                    dp[i][a][b] = min(dp[i][a][b], 
                                                     dp[i-1][a_prev][b_prev] + cost[i][a] + cost[n-1-i][b])
        
        # Find the minimum cost among all valid final states
        min_cost = float('inf')
        for a in range(3):
            for b in range(3):
                if a != b:
                    min_cost = min(min_cost, dp[n//2-1][a][b])
        
        return min_cost