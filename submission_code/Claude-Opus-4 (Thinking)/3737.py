class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        num_pairs = n // 2
        
        # dp[c1][c2] = min cost to paint the current pair with colors c1 and c2
        prev_dp = [[float('inf')] * 3 for _ in range(3)]
        
        # Base case: first pair (houses 0 and n-1)
        for c1 in range(3):
            for c2 in range(3):
                if c1 != c2:
                    prev_dp[c1][c2] = cost[0][c1] + cost[n-1][c2]
        
        # Process remaining pairs
        for i in range(1, num_pairs):
            curr_dp = [[float('inf')] * 3 for _ in range(3)]
            
            for c1 in range(3):
                for c2 in range(3):
                    if c1 != c2:  # Colors in the same pair must be different
                        for prev_c1 in range(3):
                            for prev_c2 in range(3):
                                # Check all constraints
                                if prev_c1 != prev_c2 and prev_c1 != c1 and prev_c2 != c2:
                                    curr_dp[c1][c2] = min(curr_dp[c1][c2], 
                                                         prev_dp[prev_c1][prev_c2] + cost[i][c1] + cost[n-1-i][c2])
            
            prev_dp = curr_dp
        
        # Find the minimum cost
        min_cost = float('inf')
        for c1 in range(3):
            for c2 in range(3):
                if c1 != c2:
                    min_cost = min(min_cost, prev_dp[c1][c2])
        
        return min_cost