class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        
        # Use memoization with (row_index, used_values_as_frozenset)
        memo = {}
        
        def dfs(row, used_values):
            if row == m:
                return 0
            
            # Convert set to frozenset for hashing
            key = (row, frozenset(used_values))
            if key in memo:
                return memo[key]
            
            # Option 1: Skip current row
            max_score = dfs(row + 1, used_values)
            
            # Option 2: Select a cell from current row
            for col in range(len(grid[row])):
                val = grid[row][col]
                if val not in used_values:
                    new_used = used_values | {val}
                    score = val + dfs(row + 1, new_used)
                    max_score = max(max_score, score)
            
            memo[key] = max_score
            return max_score
        
        return dfs(0, set())