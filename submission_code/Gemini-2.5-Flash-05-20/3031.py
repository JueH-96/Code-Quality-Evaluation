from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        
        n = len(grid)
        m = len(grid[0])
        total_elements = n * m
        
        # Step 1: Flatten the 2D grid into a 1D list
        flat_grid = [0] * total_elements
        k = 0
        for r in range(n):
            for c in range(m):
                flat_grid[k] = grid[r][c]
                k += 1
        
        # Step 2: Compute prefix products
        # prefix_prod[i] stores the product of flat_grid[0]...flat_grid[i] (modulo MOD)
        prefix_prod = [1] * total_elements
        # total_elements is guaranteed to be >= 2 by constraints (2 <= n*m)
        prefix_prod[0] = flat_grid[0] % MOD
        for i in range(1, total_elements):
            prefix_prod[i] = (prefix_prod[i-1] * flat_grid[i]) % MOD
        
        # Step 3: Compute suffix products
        # suffix_prod[i] stores the product of flat_grid[i]...flat_grid[total_elements-1] (modulo MOD)
        suffix_prod = [1] * total_elements
        suffix_prod[total_elements - 1] = flat_grid[total_elements - 1] % MOD
        for i in range(total_elements - 2, -1, -1):
            suffix_prod[i] = (suffix_prod[i+1] * flat_grid[i]) % MOD
        
        # Step 4: Calculate the result for each element in the flattened list
        result_flat = [0] * total_elements
        for i in range(total_elements):
            left_product = 1
            right_product = 1
            
            # If there are elements to the left, take their product from prefix_prod
            if i > 0:
                left_product = prefix_prod[i-1]
            
            # If there are elements to the right, take their product from suffix_prod
            if i < total_elements - 1:
                right_product = suffix_prod[i+1]
            
            result_flat[i] = (left_product * right_product) % MOD
            
        # Step 5: Reshape the 1D result back into a 2D matrix
        p = [[0] * m for _ in range(n)]
        k = 0
        for r in range(n):
            for c in range(m):
                p[r][c] = result_flat[k]
                k += 1
                
        return p