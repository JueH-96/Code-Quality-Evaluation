from typing import List
from functools import lru_cache

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        sorted_cols = [sorted(col, reverse=True) for col in zip(*grid)]
        
        @lru_cache(None)
        def dp(row_mask, col_idx):
            if col_idx == cols or row_mask == 0:
                return 0
            
            max_score = dp(row_mask, col_idx + 1)
            for row in range(rows):
                if not (row_mask & (1 << row)):
                    next_mask = row_mask | (1 << row)
                    score = sorted_cols[col_idx][row] + dp(next_mask, col_idx + 1)
                    max_score = max(max_score, score)
            
            return max_score
        
        return dp(0, 0)