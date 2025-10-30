from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # Calculate flips needed to make all rows palindromic
        row_flips = 0
        # Iterate through each row
        for r in range(m):
            # Iterate through the first half of the current row
            # For a row of length n, elements grid[r][j] and grid[r][n - 1 - j]
            # form a symmetric pair. We only need to check up to n // 2 - 1 because
            # iterating further would check pairs already covered or the middle element (if any),
            # which does not affect the palindrome property.
            for j in range(n // 2):
                # If the values at symmetric positions are different, one flip is needed
                # to make them the same (e.g., 0 vs 1).
                if grid[r][j] != grid[r][n - 1 - j]:
                    row_flips += 1

        # Calculate flips needed to make all columns palindromic
        col_flips = 0
        # Iterate through each column
        for c in range(n):
            # Iterate through the first half of the current column
            # For a column of height m, elements grid[i][c] and grid[m - 1 - i][c]
            # form a symmetric pair. We check up to m // 2 - 1.
            for i in range(m // 2):
                # If the values at symmetric positions are different, one flip is needed.
                if grid[i][c] != grid[m - 1 - i][c]:
                    col_flips += 1
        
        # The problem asks for the minimum number of flips to make *either* all rows
        # palindromic *or* all columns palindromic. So we return the minimum of the two costs.
        return min(row_flips, col_flips)