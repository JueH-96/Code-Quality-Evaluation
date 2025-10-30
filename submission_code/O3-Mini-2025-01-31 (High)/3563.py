from typing import List
from functools import lru_cache

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        
        # Use DFS over rows with memoization.
        # "used" is a frozenset of numbers already selected so far.
        @lru_cache(maxsize=None)
        def dfs(row: int, used: frozenset) -> int:
            # Base case: if we've processed all rows, the sum is 0.
            if row == m:
                return 0
            # Option 1: Skip the current row.
            best = dfs(row + 1, used)
            # Option 2: Try selecting a cell from the current row.
            for val in grid[row]:
                if val not in used:
                    # Add the current value and update used set.
                    best = max(best, val + dfs(row + 1, used | frozenset((val,))))
            return best
        
        return dfs(0, frozenset())