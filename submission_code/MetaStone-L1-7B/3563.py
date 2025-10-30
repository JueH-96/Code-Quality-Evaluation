from functools import lru_cache
from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        @lru_cache(maxsize=None)
        def backtrack(row, used):
            if row == len(grid):
                return 0
            max_sum = 0
            unique_vals = list(set(grid[row]))
            for val in unique_vals:
                if val not in used:
                    new_used = frozenset(used | {val})
                    res = backtrack(row + 1, new_used)
                    if res > max_sum:
                        max_sum = res
            return max_sum
        
        return backtrack(0, frozenset())