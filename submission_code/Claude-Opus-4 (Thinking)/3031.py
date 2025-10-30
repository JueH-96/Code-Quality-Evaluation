class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Flatten the grid
        flat = [grid[i][j] for i in range(n) for j in range(m)]
        total = len(flat)
        
        # Compute prefix products
        prefix = [1] * total
        for i in range(1, total):
            prefix[i] = (prefix[i-1] * flat[i-1]) % MOD
        
        # Compute suffix products
        suffix = [1] * total
        for i in range(total - 2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % MOD
        
        # Compute result and reshape to 2D
        result = [[0] * m for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                result[i][j] = (prefix[idx] * suffix[idx]) % MOD
                idx += 1
        
        return result