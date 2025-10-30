class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0

        n = len(grid)
        m = len(grid[0])

        row_counts = [0] * n
        col_counts = [0] * m

        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    row_counts[r] += 1
                    col_counts[c] += 1

        total = 0
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    R = row_counts[r] - 1
                    C = col_counts[c] - 1
                    total += R * C

        return total