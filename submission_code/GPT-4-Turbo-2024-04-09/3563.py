class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from functools import lru_cache
        
        # Number of rows and columns
        R = len(grid)
        C = len(grid[0])
        
        # Convert each row to a set of (value, column_index) tuples
        row_values = [set((grid[r][c], c) for c in range(C)) for r in range(R)]
        
        @lru_cache(None)
        def dp(row, used_columns):
            if row == R:
                return 0
            
            max_score = 0
            for value, col in row_values[row]:
                if not (used_columns & (1 << col)):  # Check if this column is not used
                    # Include this cell and move to the next row
                    max_score = max(max_score, value + dp(row + 1, used_columns | (1 << col)))
            return max_score
        
        return dp(0, 0)