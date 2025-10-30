class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        n = len(grid)
        if n == 0:
            return []
        m = len(grid[0])
        total_elements = n * m
        
        # Flatten the grid into a 1D list
        flattened = []
        for row in grid:
            flattened.extend(row)
        
        # Compute prefix products
        prefix = [1] * total_elements
        for i in range(1, total_elements):
            prefix[i] = (prefix[i-1] * flattened[i-1]) % mod
        
        # Compute suffix products
        suffix = [1] * total_elements
        for i in range(total_elements - 2, -1, -1):
            suffix[i] = (suffix[i+1] * flattened[i+1]) % mod
        
        # Calculate the result for each element
        result_flat = [(prefix[i] * suffix[i]) % mod for i in range(total_elements)]
        
        # Reshape the result back into the original grid structure
        result = []
        for i in range(n):
            start = i * m
            end = start + m
            result.append(result_flat[start:end])
        
        return result