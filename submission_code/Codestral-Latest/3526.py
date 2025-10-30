class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def is_palindromic(arr):
            return arr == arr[::-1]

        def count_flips_to_palindromic(arr):
            n = len(arr)
            flips = 0
            for i in range(n // 2):
                if arr[i] != arr[n - i - 1]:
                    flips += 1
            return flips

        m, n = len(grid), len(grid[0])

        # Calculate the minimum flips to make all rows palindromic
        min_flips_rows = float('inf')
        for row in grid:
            flips = count_flips_to_palindromic(row)
            min_flips_rows = min(min_flips_rows, flips)

        # Calculate the minimum flips to make all columns palindromic
        min_flips_cols = float('inf')
        for col in range(n):
            col_arr = [grid[row][col] for row in range(m)]
            flips = count_flips_to_palindromic(col_arr)
            min_flips_cols = min(min_flips_cols, flips)

        # Return the minimum of the two
        return min(min_flips_rows, min_flips_cols)