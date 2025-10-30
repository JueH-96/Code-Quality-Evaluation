class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        if n == 0:
            return 0
        
        row_counts = [sum(row) for row in grid]
        col_counts = [0] * n
        
        for j in range(n):
            for i in range(m):
                if grid[i][j] == 1:
                    col_counts[j] += 1
        
        total = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return total