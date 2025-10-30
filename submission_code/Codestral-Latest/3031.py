class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        MOD = 12345

        # Initialize the product matrix with ones
        product_matrix = [[1] * m for _ in range(n)]

        # Calculate the product of all elements in the grid
        total_product = 1
        for i in range(n):
            for j in range(m):
                total_product = (total_product * grid[i][j]) % MOD

        # Calculate the product matrix
        for i in range(n):
            for j in range(m):
                product_matrix[i][j] = (total_product * pow(grid[i][j], MOD - 2, MOD)) % MOD

        return product_matrix