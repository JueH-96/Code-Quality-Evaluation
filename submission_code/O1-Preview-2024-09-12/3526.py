class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # Function to compute flips needed to make all rows palindromic
        def flips_for_rows():
            total_flips = 0
            for row in grid:
                flips = 0
                for i in range(len(row) // 2):
                    if row[i] != row[~i]:
                        flips += 1
                total_flips += flips
            return total_flips

        # Function to compute flips needed to make all columns palindromic
        def flips_for_cols():
            total_flips = 0
            for c in range(n):
                flips = 0
                for i in range(m // 2):
                    if grid[i][c] != grid[~i][c]:
                        flips +=1
                total_flips += flips
            return total_flips

        min_flips = min(flips_for_rows(), flips_for_cols())
        return min_flips