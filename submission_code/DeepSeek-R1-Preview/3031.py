class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        # Flatten the grid into a 1D list
        flat = [num for row in grid for num in row]
        n = len(flat)
        if n == 0:
            return []
        
        # Compute prefix products
        prefix = [1] * n
        for i in range(1, n):
            prefix[i] = (prefix[i-1] * flat[i-1]) % MOD
        
        # Compute suffix products
        suffix = [1] * n
        for i in range(n-2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % MOD
        
        # Compute the product array
        product_flat = [(prefix[i] * suffix[i]) % MOD for i in range(n)]
        
        # Reshape the product array back into the original grid dimensions
        n_rows = len(grid)
        m_cols = len(grid[0]) if n_rows > 0 else 0
        product = []
        for i in range(n_rows):
            start = i * m_cols
            end = start + m_cols
            product.append(product_flat[start:end])
        
        return product