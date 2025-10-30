class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        
        # Flatten the grid
        flat = [grid[i][j] for i in range(n) for j in range(m)]
        
        total = len(flat)
        MOD = 12345
        
        # Compute prefix products
        prefix = [1] * total
        for i in range(1, total):
            prefix[i] = (prefix[i-1] * flat[i-1]) % MOD
        
        # Compute suffix products
        suffix = [1] * total
        for i in range(total-2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % MOD
        
        # Compute the result
        result = []
        idx = 0
        for i in range(n):
            row = []
            for j in range(m):
                product = (prefix[idx] * suffix[idx]) % MOD
                row.append(product)
                idx += 1
            result.append(row)
        
        return result