from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        mod = 12345
        total_product = 1
        zero_count = 0
        
        # Calculate the total product and count zeros
        for row in grid:
            for val in row:
                if val == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * val) % mod
        
        # If there are more than one zero, all elements in the product matrix will be zero
        if zero_count > 1:
            return [[0 for _ in range(m)] for _ in range(n)]
        
        # Construct the product matrix
        product_matrix = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if zero_count == 1:
                    if grid[i][j] == 0:
                        product_matrix[i][j] = total_product
                    else:
                        product_matrix[i][j] = 0
                else:
                    product_matrix[i][j] = (total_product * pow(grid[i][j], mod - 2, mod)) % mod
        
        return product_matrix