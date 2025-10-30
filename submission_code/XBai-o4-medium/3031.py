from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        n = len(grid)
        if n == 0:
            return []
        m = len(grid[0])
        k = n * m
        nums = []
        for row in grid:
            nums.extend(row)
        
        # Compute prefix products
        prefix = [1] * k
        for i in range(1, k):
            prefix[i] = (prefix[i-1] * nums[i-1]) % mod
        
        # Compute suffix products
        suffix = [1] * k
        for i in range(k-2, -1, -1):
            suffix[i] = (suffix[i+1] * nums[i+1]) % mod
        
        # Calculate the product array
        product_array = [(prefix[i] * suffix[i]) % mod for i in range(k)]
        
        # Reshape into original grid dimensions
        result = []
        idx = 0
        for i in range(n):
            row = []
            for j in range(m):
                row.append(product_array[idx])
                idx += 1
            result.append(row)
        
        return result