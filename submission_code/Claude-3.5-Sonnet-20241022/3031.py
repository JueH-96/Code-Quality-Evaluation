class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Flatten the grid into 1D array for easier processing
        flat = []
        for row in grid:
            flat.extend(row)
        
        # Calculate prefix products from left to right
        prefix = [1] * (n * m)
        curr = 1
        for i in range(n * m):
            prefix[i] = curr
            curr = (curr * flat[i]) % MOD
            
        # Calculate suffix products from right to left
        suffix = [1] * (n * m)
        curr = 1
        for i in range(n * m - 1, -1, -1):
            suffix[i] = curr
            curr = (curr * flat[i]) % MOD
            
        # Calculate final result
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                idx = i * m + j
                # Multiply prefix and suffix for each position
                prod = (prefix[idx] * suffix[idx]) % MOD
                row.append(prod)
            result.append(row)
            
        return result