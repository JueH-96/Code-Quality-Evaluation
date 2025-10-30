class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        min_row, max_row, min_col, max_col = float('inf'), -1, float('inf'), -1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        if min_row == float('inf'):
            return 0
        
        return (max_row - min_row + 1) * (max_col - min_col + 1)