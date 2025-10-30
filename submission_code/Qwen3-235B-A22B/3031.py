from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        # Flatten the grid into a 1D array
        arr = [num for row in grid for num in row]
        k = len(arr)
        
        # Compute prefix products mod 12345
        prefix = [1] * (k + 1)
        for i in range(1, k + 1):
            prefix[i] = (prefix[i - 1] * arr[i - 1]) % mod
        
        # Compute suffix products mod 12345
        suffix = [1] * (k + 1)
        for i in range(k - 1, -1, -1):
            suffix[i] = (suffix[i + 1] * arr[i]) % mod
        
        # Calculate the product for each element
        product = []
        for i in range(k):
            product.append((prefix[i] * suffix[i + 1]) % mod)
        
        # Reshape the product list back into the original grid structure
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        result = []
        for i in range(n):
            start = i * m
            end = start + m
            result.append(product[start:end])
        
        return result