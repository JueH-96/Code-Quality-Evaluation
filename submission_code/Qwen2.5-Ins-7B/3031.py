from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        product_matrix = [[0] * m for _ in range(n)]
        prefix_product = 1
        
        # Calculate prefix product for each row
        for i in range(n):
            for j in range(m):
                product_matrix[i][j] = prefix_product
                prefix_product = (prefix_product * grid[i][j]) % 12345
        
        # Calculate suffix product for each row
        suffix_product = 1
        for i in range(n-1, -1, -1):
            for j in range(m-1, -1, -1):
                product_matrix[i][j] = (product_matrix[i][j] * suffix_product) % 12345
                suffix_product = (suffix_product * grid[i][j]) % 12345
        
        return product_matrix