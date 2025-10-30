from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        # Flatten the grid into a 1D list
        n = len(grid)
        m = len(grid[0])
        K = n * m
        arr = [0] * K
        for i in range(n):
            for j in range(m):
                arr[i * m + j] = grid[i][j] % MOD
        
        # Build prefix products: prefix[i] = product of arr[0..i-1] mod MOD
        prefix = [1] * (K + 1)
        for i in range(K):
            prefix[i+1] = (prefix[i] * arr[i]) % MOD
        
        # Build suffix products: suffix[i] = product of arr[i..K-1] mod MOD
        suffix = [1] * (K + 1)
        for i in range(K - 1, -1, -1):
            suffix[i] = (suffix[i+1] * arr[i]) % MOD
        
        # Construct the result matrix
        result = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # product of all except arr[idx] = prefix[idx] * suffix[idx+1]
                result[i][j] = (prefix[idx] * suffix[idx+1]) % MOD
        
        return result