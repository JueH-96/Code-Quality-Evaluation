class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        total_product = 1
        for i in range(n):
            for j in range(m):
                total_product = (total_product * grid[i][j]) % 12345
        
        result = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if total_product == 0:
                    if grid[i][j] == 0:
                        temp_product = 1
                        for row in range(n):
                            for col in range(m):
                                if row != i or col != j:
                                    temp_product = (temp_product * grid[row][col]) % 12345
                        result[i][j] = temp_product
                    else:
                        result[i][j] = 0
                else:
                    result[i][j] = (total_product * pow(grid[i][j], 12345 - 2, 12345)) % 12345
        return result