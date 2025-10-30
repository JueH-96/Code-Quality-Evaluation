class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        flat_grid = []
        for i in range(n):
            for j in range(m):
                flat_grid.append(grid[i][j])
        N = n * m
        prefix_prod = [1] * N
        suffix_prod = [1] * N
        
        if N > 0:
            prefix_prod[0] = 1
            for k in range(1, N):
                prefix_prod[k] = (prefix_prod[k-1] * flat_grid[k-1]) % 12345
                
            suffix_prod[N-1] = 1
            for k in range(N-2, -1, -1):
                suffix_prod[k] = (suffix_prod[k+1] * flat_grid[k+1]) % 12345
                
        p = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                k = i * m + j
                p[i][j] = (prefix_prod[k] * suffix_prod[k]) % 12345
                
        return p