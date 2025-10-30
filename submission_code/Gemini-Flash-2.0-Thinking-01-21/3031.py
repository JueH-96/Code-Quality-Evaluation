class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        p = [[0] * m for _ in range(n)]
        mod = 12345
        for i in range(n):
            for j in range(m):
                product = 1
                for r in range(n):
                    for c in range(m):
                        if r != i or c != j:
                            product = (product * grid[r][c]) % mod
                p[i][j] = product
        return p