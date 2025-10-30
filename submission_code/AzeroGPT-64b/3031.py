from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        if len(grid) * len(grid[0]) == 0:
            return []
        
        row_number = len(grid)
        col_number = len(grid[0])
        prod_left = [1] * col_number
        prod_right = [1] * col_number
        
        # Calculate left-side multiplications
        for r in range(row_number):
            for c in range(1, col_number):
                prod_left[c] = (prod_left[c - 1] * grid[r][c - 1]) % mod
                
        # Calculate right-side multiplications with result
        for r in range(row_number):
            for c in range(col_number - 2, -1, -1):
                prod_right[c] = (prod_right[c + 1] * grid[r][c + 1]) % mod
                
            for c in range(col_number):
                grid[r][c] = (prod_left[c] * prod_right[c]) % mod
                
        return grid