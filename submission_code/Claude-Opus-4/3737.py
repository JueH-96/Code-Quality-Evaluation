class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # Number of pairs
        pairs = n // 2
        
        # Initialize DP for the first pair
        # dp[j][k] = minimum cost to paint first pair with colors j and k
        prev_dp = {}
        for j in range(3):
            for k in range(3):
                if j != k:  # Equidistant houses must have different colors
                    prev_dp[(j, k)] = cost[0][j] + cost[n-1][k]
        
        # Process remaining pairs
        for i in range(1, pairs):
            curr_dp = {}
            
            for j in range(3):
                for k in range(3):
                    if j != k:  # Current pair must have different colors
                        min_cost = float('inf')
                        
                        # Try all valid previous pair colors
                        for prev_j in range(3):
                            for prev_k in range(3):
                                if prev_j != prev_k and prev_j != j and prev_k != k:
                                    # Check adjacency constraints
                                    # prev_j is color of house[i-1], j is color of house[i]
                                    # prev_k is color of house[n-i], k is color of house[n-1-i]
                                    if (prev_j, prev_k) in prev_dp:
                                        min_cost = min(min_cost, prev_dp[(prev_j, prev_k)])
                        
                        if min_cost != float('inf'):
                            curr_dp[(j, k)] = min_cost + cost[i][j] + cost[n-1-i][k]
            
            prev_dp = curr_dp
        
        # Find minimum cost among all valid final configurations
        return min(prev_dp.values())