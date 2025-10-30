from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Calculate the total product of all elements in the grid
        total_product = 1
        zero_count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * grid[i][j]) % MOD
        
        # If there are two or more zeros, the product of all other elements will be zero
        if zero_count >= 2:
            return [[0] * m for _ in range(n)]
        
        # If there is one zero, the product of all other elements will be the total product
        if zero_count == 1:
            result = [[0] * m for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    if grid[i][j] == 0:
                        result[i][j] = total_product
            return result
        
        # If there are no zeros, calculate the product of all other elements for each cell
        result = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                result[i][j] = (total_product * pow(grid[i][j], MOD - 2, MOD)) % MOD
        
        return result