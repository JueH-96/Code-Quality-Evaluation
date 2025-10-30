class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        import numpy as np
        
        n = len(grid)
        m = len(grid[0])
        
        # Calculate the total product of all elements in the grid
        total_product = 1
        for row in grid:
            for value in row:
                total_product *= value
                total_product %= 12345  # Keep it manageable with modulo
        
        # Construct the product matrix
        product_matrix = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                # Calculate the product excluding grid[i][j]
                product_matrix[i][j] = (total_product * pow(grid[i][j], -1, 12345)) % 12345
        
        return product_matrix