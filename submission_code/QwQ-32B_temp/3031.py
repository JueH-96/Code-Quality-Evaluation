from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        # Flatten the grid into a 1D list
        elements = []
        for row in grid:
            elements.extend(row)
        n = len(elements)
        if n == 0:
            return []
        
        # Compute left products
        left = [1] * n
        for i in range(1, n):
            left[i] = (left[i-1] * elements[i-1]) % MOD
        
        # Compute right products
        right = [1] * n
        for i in range(n-2, -1, -1):
            right[i] = (right[i+1] * elements[i+1]) % MOD
        
        # Calculate the product for each element
        products = [(left[i] * right[i]) % MOD for i in range(n)]
        
        # Reshape the products back into the original grid structure
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        result = []
        idx = 0
        for r in range(rows):
            new_row = products[idx : idx + cols]
            result.append(new_row)
            idx += cols
        
        return result