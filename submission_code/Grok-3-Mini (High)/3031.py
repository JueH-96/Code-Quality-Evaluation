import math
from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])
        N = n * m
        
        # Flatten the grid and take modulo MOD
        modded_nums = [val % MOD for row in grid for val in row]
        
        # Compute prefix products
        prefix = [0] * (N + 1)
        prefix[0] = 1
        for k in range(1, N + 1):
            prefix[k] = (prefix[k - 1] * modded_nums[k - 1] % MOD) % MOD
        
        # Compute suffix products
        suffix = [0] * (N + 1)
        suffix[N] = 1
        for k in range(N - 1, -1, -1):
            suffix[k] = (modded_nums[k] * suffix[k + 1] % MOD) % MOD
        
        # Create result grid and compute the product excluding each element
        result = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                idx = i * m + j  # Flat index
                prod_excl = (prefix[idx] * suffix[idx + 1] % MOD) % MOD
                result[i][j] = prod_excl
        
        return result