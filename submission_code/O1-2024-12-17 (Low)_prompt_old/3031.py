class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        # Modulus
        MOD = 12345
        
        # Flatten the 2D grid into a 1D array for convenience
        n = len(grid)
        m = len(grid[0])
        arr = []
        for row in grid:
            arr.extend(row)
        
        length = n * m
        
        # Compute prefix products (mod MOD)
        prefix_prod = [0] * length
        prefix_prod[0] = 1
        for i in range(1, length):
            prefix_prod[i] = (prefix_prod[i-1] * (arr[i-1] % MOD)) % MOD
        
        # Compute suffix products (mod MOD)
        suffix_prod = [0] * length
        suffix_prod[length - 1] = 1
        for i in range(length - 2, -1, -1):
            suffix_prod[i] = (suffix_prod[i+1] * (arr[i+1] % MOD)) % MOD
        
        # Build the result matrix
        result = [[0]*m for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                # Calculate product of all except arr[idx] = prefix_prod[idx] * suffix_prod[idx] (mod MOD)
                result[i][j] = (prefix_prod[idx] * suffix_prod[idx]) % MOD
                idx += 1
        
        return result