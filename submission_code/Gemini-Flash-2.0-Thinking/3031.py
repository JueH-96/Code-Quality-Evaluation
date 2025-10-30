class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        mod = 12345
        num_zeros = 0
        zero_row, zero_col = -1, -1

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    num_zeros += 1
                    zero_row, zero_col = i, j

        if num_zeros > 1:
            return [[0 for _ in range(m)] for _ in range(n)]
        elif num_zeros == 1:
            p = [[0 for _ in range(m)] for _ in range(n)]
            product_non_zero = 1
            for i in range(n):
                for j in range(m):
                    if grid[i][j] != 0:
                        product_non_zero = (product_non_zero * grid[i][j]) % mod
            p[zero_row][zero_col] = product_non_zero
            return p
        else:
            total_product = 1
            for i in range(n):
                for j in range(m):
                    total_product = (total_product * grid[i][j]) % mod

            p = [[0 for _ in range(m)] for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    inv = pow(grid[i][j], mod - 2, mod)
                    p[i][j] = (total_product * inv) % mod
            return p