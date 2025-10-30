from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def count_flips_to_make_palindromic(arr):
            flips = 0
            i, j = 0, len(arr) - 1
            while i < j:
                if arr[i] != arr[j]:
                    flips += 1
                i += 1
                j -= 1
            return flips

        row_flips = sum(count_flips_to_make_palindromic(row) for row in grid)

        col_flips = sum(count_flips_to_make_palindromic([grid[i][j] for i in range(m)]) for j in range(n))

        return min(row_flips, col_flips)