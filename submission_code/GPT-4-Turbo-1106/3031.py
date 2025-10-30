from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n, m = len(grid), len(grid[0])
        
        # Calculate the row and column products
        row_products = [1] * n
        col_products = [1] * m
        
        for i in range(n):
            for j in range(m):
                row_products[i] = (row_products[i] * grid[i][j]) % MOD
                col_products[j] = (col_products[j] * grid[i][j]) % MOD
        
        # Calculate the total product of all elements
        total_product = 1
        for rp in row_products:
            total_product = (total_product * rp) % MOD
        
        # Construct the product matrix
        product_matrix = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                # Exclude the current element from the product
                if n == 1 or m == 1:  # Special case for single row or column
                    product_matrix[i][j] = (total_product * pow(grid[i][j], MOD - 2, MOD)) % MOD
                else:
                    product_matrix[i][j] = (total_product * pow(row_products[i] * col_products[j], MOD - 2, MOD)) % MOD
                product_matrix[i][j] = (product_matrix[i][j] * pow(grid[i][j], MOD - 2, MOD)) % MOD
        
        return product_matrix