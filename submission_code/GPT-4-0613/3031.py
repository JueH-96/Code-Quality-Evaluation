from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        total_product = 1
        zero_count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * grid[i][j]) % 12345

        product_matrix = [[0]*m for _ in range(n)]
        if zero_count > 1:
            return product_matrix
        elif zero_count == 1:
            for i in range(n):
                for j in range(m):
                    if grid[i][j] == 0:
                        product_matrix[i][j] = total_product
            return product_matrix
        else:
            for i in range(n):
                for j in range(m):
                    product_matrix[i][j] = (total_product * pow(grid[i][j], 12345-2, 12345)) % 12345
            return product_matrix