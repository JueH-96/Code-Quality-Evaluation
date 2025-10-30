class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Calculate the product of all elements in the grid
        total_product = 1
        for row in grid:
            for num in row:
                total_product = (total_product * num) % MOD
        
        # Calculate the product matrix
        product_matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                if grid[i][j] == 0:
                    row.append(total_product)
                else:
                    row.append((total_product // grid[i][j]) % MOD)
            product_matrix.append(row)
        
        return product_matrix