class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        product_matrix = [[0]*m for _ in range(n)]
        
        # Calculate the product of all elements in grid except for the element grid[i][j]
        for i in range(n):
            for j in range(m):
                if i == 0:
                    product_matrix[i][j] = grid[i][j] * grid[i+1][j] if j < m-1 else grid[i][j]
                elif i == n-1:
                    product_matrix[i][j] = grid[i][j] * grid[i-1][j] if j < m-1 else grid[i][j]
                else:
                    product_matrix[i][j] = grid[i][j] * grid[i-1][j] * grid[i+1][j] if j < m-1 else grid[i][j]
        
        # Take modulo 12345
        for i in range(n):
            for j in range(m):
                product_matrix[i][j] %= 12345
        
        return product_matrix