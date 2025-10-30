class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        row_product = [1] * n
        col_product = [1] * m
        for i in range(n):
            for j in range(m):
                row_product[i] = (row_product[i] * grid[i][j]) % 12345
                col_product[j] = (col_product[j] * grid[i][j]) % 12345
        result = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                result[i][j] = (row_product[i] * col_product[j]) % 12345
        return result