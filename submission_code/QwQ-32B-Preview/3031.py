from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        # Flatten the grid into 1D
        flat = [elem for row in grid for elem in row]
        n = len(grid)
        m = len(grid[0])
        size = n * m
        MOD = 12345
        
        # Compute prefix products
        prefix = [1] * size
        for k in range(1, size):
            prefix[k] = prefix[k-1] * flat[k-1] % MOD
        
        # Compute suffix products
        suffix = [1] * size
        for k in range(size-2, -1, -1):
            suffix[k] = suffix[k+1] * flat[k+1] % MOD
        
        # Compute prod array
        prod = [(prefix[k] * suffix[k]) % MOD for k in range(size)]
        
        # Reshape prod back into 2D grid
        p = []
        idx = 0
        for i in range(n):
            row = []
            for j in range(m):
                row.append(prod[idx])
                idx += 1
            p.append(row)
        
        return p