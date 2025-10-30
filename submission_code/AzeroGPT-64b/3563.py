from typing import List
from functools import lru_cache

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        
        selected_cols = [0] * cols
        for i in range(rows):
            max_in_col = [0] * cols
            for j in range(cols):
                max_in_col[j] = max(max_in_col[j], grid[i][j] if selected_cols[j] == 0 else 0)
                if j > 0:
                    max_in_col[j] = max(max_in_col[j], max_in_col[j-1])
            selected_cols = max_in_col
        
        @lru_cache(None)
        def dp(row, mask):
            if row == rows:
                return 0
            
            max_score = 0
            for col in range(cols):
                if mask & (1 << col) == 0 and grid[row][col] > selected_cols[col]:
                    new_mask = mask | (1 << col)
                    score = grid[row][col] + dp(row+1, new_mask)
                    max_score = max(max_score, score)
                    
            return max_score
        
        return dp(0, 0)