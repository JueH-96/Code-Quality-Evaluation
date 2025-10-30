class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        
        memo = {}
        
        def dp(row, used_values, has_selected):
            if row == m:
                return 0 if has_selected else float('-inf')
            
            key = (row, frozenset(used_values), has_selected)
            if key in memo:
                return memo[key]
            
            # Option 1: Skip this row
            result = dp(row + 1, used_values, has_selected)
            
            # Option 2: Pick a cell from this row
            for col in range(n):
                val = grid[row][col]
                if val not in used_values:
                    new_used = used_values | {val}
                    result = max(result, val + dp(row + 1, new_used, True))
            
            memo[key] = result
            return result
        
        return dp(0, set(), False)