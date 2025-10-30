class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from functools import lru_cache
        
        # Convert each row to a set to avoid duplicate values
        row_sets = [set(row) for row in grid]
        
        @lru_cache(maxsize=None)
        def solve(row, used_mask):
            if row == len(grid):
                return 0
            
            # Option 1: Skip this row
            max_score = solve(row + 1, used_mask)
            
            # Option 2: Select a value from this row
            for val in row_sets[row]:
                bit_pos = val - 1  # Values are 1-indexed, bits are 0-indexed
                if not (used_mask & (1 << bit_pos)):
                    new_mask = used_mask | (1 << bit_pos)
                    score = val + solve(row + 1, new_mask)
                    max_score = max(max_score, score)
            
            return max_score
        
        return solve(0, 0)