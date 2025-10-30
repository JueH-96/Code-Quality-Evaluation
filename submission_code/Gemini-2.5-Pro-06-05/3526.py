from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        """
        Calculates the minimum number of flips to make either all rows or all columns palindromic.
        """
        m = len(grid)
        n = len(grid[0])

        # Scenario 1: Calculate flips to make all rows palindromic.
        # For each row, we count the number of pairs of symmetric elements 
        # (from left and right) that are different. Each different pair 
        # requires one flip to be made identical.
        row_flips = 0
        for i in range(m):
            # The integer division `n // 2` correctly handles both even and odd lengths.
            for j in range(n // 2):
                if grid[i][j] != grid[i][n - 1 - j]:
                    row_flips += 1
        
        # Scenario 2: Calculate flips to make all columns palindromic.
        # The logic is analogous to the row calculation. We count mismatches
        # in symmetric pairs from top and bottom for each column.
        col_flips = 0
        for j in range(n):
            # The integer division `m // 2` correctly handles both even and odd lengths.
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_flips += 1
                    
        # The overall minimum number of flips is the minimum of the two scenarios.
        return min(row_flips, col_flips)