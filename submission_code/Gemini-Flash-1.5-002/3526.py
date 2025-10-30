class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def count_flips(arr):
            count = 0
            for i in range(len(arr) // 2):
                count += arr[i] ^ arr[len(arr) - 1 - i]
            return count

        min_flips = float('inf')

        # Check rows
        row_flips = 0
        for row in grid:
            row_flips += count_flips(row)
        min_flips = min(min_flips, row_flips)

        # Check columns
        col_flips = 0
        for j in range(n):
            col = [grid[i][j] for i in range(m)]
            col_flips += count_flips(col)
        min_flips = min(min_flips, col_flips)

        return min_flips