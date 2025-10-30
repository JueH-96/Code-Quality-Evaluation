class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        for row in grid:
            row.sort(reverse=True)
        grid.sort(key=lambda x: x[0], reverse=True)
        return sum(grid[i][i] for i in range(min(n, m)))