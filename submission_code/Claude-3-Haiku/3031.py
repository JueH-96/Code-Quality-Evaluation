class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        product_matrix = [[1] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                product = 1
                for k in range(m):
                    if k != j:
                        product *= grid[i][k]
                product_matrix[i][j] = product % 12345
        
        return product_matrix