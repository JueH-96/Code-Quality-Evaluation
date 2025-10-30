class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def count_flips(arr):
            return sum(a != b for a, b in zip(arr, reversed(arr))) // 2
        
        row_flips = sum(count_flips(row) for row in grid)
        col_flips = sum(count_flips([grid[i][j] for i in range(m)]) for j in range(n))
        
        return min(row_flips, col_flips)