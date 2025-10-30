from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        product_matrix = [[0] * cols for _ in range(rows)]
        
        # Calculate prefix products
        prefix = [1]
        for row in grid:
            for val in row:
                prefix.append((prefix[-1] * val) % 12345)
        
        # Calculate suffix products
        suffix = [1]
        for row in reversed(grid):
            for val in reversed(row):
                suffix.append((suffix[-1] * val) % 12345)
        suffix = list(reversed(suffix))
        
        # Calculate product matrix
        for i in range(rows):
            for j in range(cols):
                left = prefix[i * cols + j]
                right = suffix[(i * cols + j + 1) * 2]
                product_matrix[i][j] = (left * right) % 12345
        
        return product_matrix