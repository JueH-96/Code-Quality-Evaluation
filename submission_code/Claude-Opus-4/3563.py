class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        # Get all unique values and their positions
        value_positions = {}
        for i in range(m):
            for j in range(n):
                val = grid[i][j]
                if val not in value_positions:
                    value_positions[val] = []
                value_positions[val].append(i)
        
        # Sort values in descending order to prioritize higher values
        unique_values = sorted(value_positions.keys(), reverse=True)
        
        # dp[mask] = maximum score using rows indicated by mask
        dp = {}
        dp[0] = 0
        
        for val in unique_values:
            new_dp = dp.copy()
            rows = value_positions[val]
            
            # Try to assign this value to one of its available rows
            for row in rows:
                row_bit = 1 << row
                
                # Check all previous states
                for mask, score in dp.items():
                    # If this row hasn't been used in this state
                    if mask & row_bit == 0:
                        new_mask = mask | row_bit
                        new_score = score + val
                        if new_mask not in new_dp or new_dp[new_mask] < new_score:
                            new_dp[new_mask] = new_score
            
            dp = new_dp
        
        return max(dp.values())