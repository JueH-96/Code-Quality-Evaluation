from typing import List

class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        # Calculate flips to make all rows palindromic
        row_flips = 0
        for r_idx in range(m):
            # Current row is grid[r_idx], which has length n.
            # We need to compare elements grid[r_idx][c_idx] and grid[r_idx][n - 1 - c_idx]
            # for c_idx from 0 up to (n // 2) - 1.
            # The loop range(n // 2) correctly iterates c_idx from 0 to (n // 2) - 1.
            for c_idx in range(n // 2): 
                if grid[r_idx][c_idx] != grid[r_idx][n - 1 - c_idx]:
                    row_flips += 1
        
        # Calculate flips to make all columns palindromic
        col_flips = 0
        for c_idx in range(n):
            # Current column consists of elements grid[r_idx][c_idx] for r_idx from 0 to m-1.
            # This column has length m.
            # We need to compare elements grid[r_idx][c_idx] and grid[m - 1 - r_idx][c_idx]
            # for r_idx from 0 up to (m // 2) - 1.
            # The loop range(m // 2) correctly iterates r_idx from 0 to (m // 2) - 1.
            for r_idx in range(m // 2):
                if grid[r_idx][c_idx] != grid[m - 1 - r_idx][c_idx]:
                    col_flips += 1
            
        return min(row_flips, col_flips)