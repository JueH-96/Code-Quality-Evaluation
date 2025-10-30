class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        MOD = 12345
        
        # Flatten the grid
        flat = []
        for i in range(n):
            for j in range(m):
                flat.append(grid[i][j] % MOD)
        
        total_elements = n * m
        
        # Calculate prefix products
        prefix = [1] * total_elements
        for i in range(1, total_elements):
            prefix[i] = (prefix[i-1] * flat[i-1]) % MOD
        
        # Calculate suffix products
        suffix = [1] * total_elements
        for i in range(total_elements - 2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % MOD
        
        # Build result matrix
        result = [[0] * m for _ in range(n)]
        
        idx = 0
        for i in range(n):
            for j in range(m):
                result[i][j] = (prefix[idx] * suffix[idx]) % MOD
                idx += 1
        
        return result