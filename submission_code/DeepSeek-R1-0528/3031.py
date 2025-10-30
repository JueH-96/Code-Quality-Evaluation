class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        A = []
        for row in grid:
            A.extend(row)
        
        n = len(A)
        res = [1] * n
        mod = 12345
        
        left_product = 1
        for i in range(n):
            res[i] = left_product
            left_product = (left_product * A[i]) % mod
        
        right_product = 1
        for i in range(n-1, -1, -1):
            res[i] = (res[i] * right_product) % mod
            right_product = (right_product * A[i]) % mod
        
        rows = len(grid)
        cols = len(grid[0])
        product_grid = []
        idx = 0
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(res[idx])
                idx += 1
            product_grid.append(row)
        
        return product_grid