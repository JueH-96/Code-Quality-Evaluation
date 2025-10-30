from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])
        
        # Calculate the total product of all elements in the grid
        total_product = 1
        for i in range(n):
            for j in range(m):
                total_product = (total_product * grid[i][j]) % MOD
        
        # Create the product matrix
        product_matrix = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                # Calculate the product excluding grid[i][j]
                if grid[i][j] != 0:
                    product_matrix[i][j] = (total_product * pow(grid[i][j], MOD-2, MOD)) % MOD
                else:
                    # If grid[i][j] is zero, we need to calculate the product of all other elements
                    product = 1
                    for x in range(n):
                        for y in range(m):
                            if x != i or y != j:
                                product = (product * grid[x][y]) % MOD
                    product_matrix[i][j] = product
        
        return product_matrix