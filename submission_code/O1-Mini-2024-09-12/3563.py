from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from functools import lru_cache

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        @lru_cache(None)
        def dfs(row, used):
            if row == rows:
                return 0
            max_sum = dfs(row + 1, used)  # Skip this row
            for val in grid[row]:
                if not (used & (1 << val)):
                    max_sum = max(max_sum, val + dfs(row + 1, used | (1 << val)))
            return max_sum

        return dfs(0, 0)