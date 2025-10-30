from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        # Flatten the grid into a 1D list
        flat = []
        for row in grid:
            for num in row:
                flat.append(num)
        n = len(flat)
        MOD = 12345
        
        # Compute prefix products
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = (prefix[i-1] * flat[i-1]) % MOD
        
        # Compute suffix products
        suffix = [1] * n
        for i in range(n-2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % MOD
        
        # Calculate the product for each element
        product = [(prefix[i] * suffix[i]) % MOD for i in range(n)]
        
        # Reshape the product list back into the original grid structure
        result = []
        idx = 0
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        for i in range(rows):
            new_row = []
            for j in range(cols):
                new_row.append(product[idx])
                idx += 1
            result.append(new_row)
        
        return result