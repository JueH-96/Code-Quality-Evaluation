class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m = len(grid)
        n = len(grid[0])
        row_counts = [sum(row) for row in grid]
        col_counts = [0] * n
        for j in range(n):
            col_sum = 0
            for i in range(m):
                col_sum += grid[i][j]
            col_counts[j] = col_sum
        total = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        return total