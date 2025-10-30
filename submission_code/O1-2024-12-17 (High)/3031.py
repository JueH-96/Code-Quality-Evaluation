from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])
        
        # Flatten the 2D grid into a 1D list
        arr = []
        for row in grid:
            arr.extend(row)
        
        total = n * m
        
        # Compute prefix products modulo MOD (prefix[i] = product of arr[0..i-1] mod MOD)
        prefix = [1] * total
        for i in range(1, total):
            prefix[i] = (prefix[i-1] * arr[i-1]) % MOD
        
        # Compute suffix products modulo MOD (suffix[i] = product of arr[i+1..end] mod MOD)
        suffix = [1] * total
        for i in range(total - 2, -1, -1):
            suffix[i] = (suffix[i+1] * arr[i+1]) % MOD
        
        # Compute the result for each element
        product_arr = [0] * total
        for i in range(total):
            product_arr[i] = (prefix[i] * suffix[i]) % MOD
        
        # Reshape the 1D product list back into a 2D list
        res = []
        index = 0
        for _ in range(n):
            row = []
            for _ in range(m):
                row.append(product_arr[index])
                index += 1
            res.append(row)
        
        return res