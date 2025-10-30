class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Flatten the 2D grid into a 1D array
        flat = [num for row in grid for num in row]
        total = len(flat)
        
        # Calculate prefix products
        prefix = [1] * (total + 1)
        for i in range(total):
            prefix[i + 1] = (prefix[i] * flat[i]) % MOD
        
        # Calculate suffix products
        suffix = [1] * (total + 1)
        for i in range(total - 1, -1, -1):
            suffix[i] = (suffix[i + 1] * flat[i]) % MOD
        
        # Calculate the product matrix
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                idx = i * m + j
                product = (prefix[idx] * suffix[idx + 1]) % MOD
                row.append(product)
            result.append(row)
        
        return result