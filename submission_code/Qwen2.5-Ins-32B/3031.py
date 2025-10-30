from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n, m = len(grid), len(grid[0])
        total_product = 1
        zero_count = 0
        
        # Calculate the total product of all elements in the grid
        for row in grid:
            for val in row:
                if val == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * val) % MOD
        
        # If there are more than one zero, the entire product matrix will be zero
        if zero_count > 1:
            return [[0] * m for _ in range(n)]
        
        result = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                if zero_count == 1:
                    result[i][j] = 0 if grid[i][j] != 0 else total_product
                else:
                    result[i][j] = (total_product * pow(grid[i][j], MOD-2, MOD)) % MOD
        
        return result