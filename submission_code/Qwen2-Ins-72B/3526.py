class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        row, col = [0] * m, [0] * n
        for i in range(m):
            for j in range(n // 2):
                row[i] += grid[i][j] != grid[i][n - 1 - j]
        for j in range(n):
            for i in range(m // 2):
                col[j] += grid[i][j] != grid[m - 1 - i][j]
        return min(sum(row), sum(col))