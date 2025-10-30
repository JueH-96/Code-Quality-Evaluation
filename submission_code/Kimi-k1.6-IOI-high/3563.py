class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        max_row = [max(row) for row in grid]
        remaining_max = [0] * m
        remaining_max[-1] = max_row[-1]
        for i in range(m-2, -1, -1):
            remaining_max[i] = max_row[i] + remaining_max[i+1]
        
        max_sum = 0
        
        def backtrack(row, mask, current_sum):
            nonlocal max_sum
            if row == m:
                if current_sum > max_sum:
                    max_sum = current_sum
                return
            if current_sum + remaining_max[row] <= max_sum:
                return
            backtrack(row + 1, mask, current_sum)
            for j in range(n):
                val = grid[row][j]
                if not (mask & (1 << val)):
                    new_mask = mask | (1 << val)
                    new_sum = current_sum + val
                    backtrack(row + 1, new_mask, new_sum)
        
        backtrack(0, 0, 0)
        return max_sum