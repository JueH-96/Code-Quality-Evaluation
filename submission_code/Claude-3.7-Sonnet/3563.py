class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)  # Number of rows
        
        # Memoization table
        memo = {}
        
        # Define a recursive function to calculate the maximum score
        # i: current row, picked: frozenset of already picked values
        def dp(i, picked):
            if i == m:
                return 0
            
            if (i, picked) in memo:
                return memo[(i, picked)]
            
            # Skip this row
            max_score = dp(i + 1, picked)
            
            # Try picking each cell in this row
            for j in range(len(grid[i])):
                val = grid[i][j]
                if val not in picked:
                    new_picked = picked | {val}
                    new_score = val + dp(i + 1, new_picked)
                    max_score = max(max_score, new_score)
            
            memo[(i, picked)] = max_score
            return max_score
        
        return dp(0, frozenset())