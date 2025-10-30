from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0: 
            return 0
        n = len(grid[0])
        
        # Use DFS/backtracking across rows.
        best = 0
        
        def dfs(row, current_sum, used):
            nonlocal best
            if row == m:
                best = max(best, current_sum)
                return
            
            # Option 1: Skip the row (don't choose any cell)
            dfs(row + 1, current_sum, used)
            
            # Option 2: Try to choose one cell from this row if its value is not already used.
            for col in range(n):
                val = grid[row][col]
                if val in used:
                    continue
                used.add(val)
                dfs(row + 1, current_sum + val, used)
                used.remove(val)
        
        dfs(0, 0, set())
        return best