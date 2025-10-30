class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        p = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                product = 1
                for row in range(n):
                    for col in range(m):
                        if row != i or col != j:
                            product = (product * grid[row][col]) % 12345
                p[i][j] = product
        
        return p