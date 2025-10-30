class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n, m = len(grid), len(grid[0])
        total_product = 1
        zero_count = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * grid[i][j]) % MOD
        
        result = [[0] * m for _ in range(n)]
        
        if zero_count > 1:
            return result
        
        for i in range(n):
            for j in range(m):
                if zero_count == 1:
                    if grid[i][j] == 0:
                        result[i][j] = total_product
                else:
                    result[i][j] = (total_product * pow(grid[i][j], MOD - 2, MOD)) % MOD
        
        return result