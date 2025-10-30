class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n, m = len(grid), len(grid[0])
        MOD = 12345
        
        # Flatten the grid for easier manipulation
        flat_grid = [grid[i][j] for i in range(n) for j in range(m)]
        
        # Product of elements before each position
        prefix_product = [1]
        for num in flat_grid:
            prefix_product.append((prefix_product[-1] * num) % MOD)
        
        # Product of elements after each position
        suffix_product = [1]
        for num in reversed(flat_grid):
            suffix_product.append((suffix_product[-1] * num) % MOD)
        suffix_product.reverse()
        
        # Construct the product matrix
        p = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                p[i][j] = (prefix_product[idx] * suffix_product[idx + 1]) % MOD
        
        return p