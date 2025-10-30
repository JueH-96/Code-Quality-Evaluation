class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        m = len(grid[0])
        
        # Calculate the total product and zero counts
        total_product = 1
        zero_count = 0
        zero_positions = []
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    zero_count += 1
                    zero_positions.append((i, j))
                else:
                    total_product = (total_product * grid[i][j]) % MOD
        
        # If there are more than one zero, all products will be zero
        if zero_count > 1:
            return [[0] * m for _ in range(n)]
        
        # If there is exactly one zero, only the row and column containing the zero will have non-zero products
        result = [[0] * m for _ in range(n)]
        if zero_count == 1:
            zero_i, zero_j = zero_positions[0]
            # Calculate product for the row and column of the zero
            row_product = 1
            for j in range(m):
                if j != zero_j:
                    row_product = (row_product * grid[zero_i][j]) % MOD
            col_product = 1
            for i in range(n):
                if i != zero_i:
                    col_product = (col_product * grid[i][zero_j]) % MOD
            
            for j in range(m):
                result[zero_i][j] = col_product if j != zero_j else 0
            for i in range(n):
                result[i][zero_j] = row_product if i != zero_i else 0
            
            return result
        
        # If there are no zeros, calculate the product matrix normally
        for i in range(n):
            for j in range(m):
                # Exclude grid[i][j] from total product
                result[i][j] = (total_product * pow(grid[i][j], MOD-2, MOD)) % MOD
        
        return result