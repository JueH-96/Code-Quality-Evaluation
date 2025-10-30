from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])

        # Step 1: Compute the product of all elements in the grid
        total_product = 1
        for i in range(n):
            for j in range(m):
                total_product *= grid[i][j]
                total_product %= MOD

        # Step 2: Compute the product matrix
        product_matrix = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                product_matrix[i][j] = (total_product * pow(grid[i][j], MOD-2, MOD)) % MOD

        return product_matrix