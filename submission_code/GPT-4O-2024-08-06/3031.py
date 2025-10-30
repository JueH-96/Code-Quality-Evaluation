class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])
        
        # Calculate the total product of all elements in the grid
        total_product = 1
        zero_count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] != 0:
                    total_product *= grid[i][j]
                    total_product %= MOD
                else:
                    zero_count += 1
        
        # Initialize the product matrix
        product_matrix = [[0] * m for _ in range(n)]
        
        # Fill the product matrix
        for i in range(n):
            for j in range(m):
                if zero_count > 1:
                    # If more than one zero, all products will be zero
                    product_matrix[i][j] = 0
                elif zero_count == 1:
                    # If exactly one zero, only the position of the zero will have a non-zero product
                    if grid[i][j] == 0:
                        product = 1
                        for x in range(n):
                            for y in range(m):
                                if (x != i or y != j) and grid[x][y] != 0:
                                    product *= grid[x][y]
                                    product %= MOD
                        product_matrix[i][j] = product
                    else:
                        product_matrix[i][j] = 0
                else:
                    # No zeros, normal product calculation
                    product_matrix[i][j] = (total_product * pow(grid[i][j], MOD-2, MOD)) % MOD
        
        return product_matrix