class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        row_ones = [0] * rows
        col_ones = [0] * cols
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_ones[r] += 1
                    col_ones[c] += 1
        
        ans = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    ans += (row_ones[r] - 1) * (col_ones[c] - 1)
                    
        return ans