class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        total_product = 1
        zero_count = 0
        
        # Calculate the total product of all elements and count zeros
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    zero_count += 1
                else:
                    total_product = (total_product * grid[i][j]) % 12345
        
        p = [[0 for _ in range(m)] for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                if zero_count > 1:
                    p[i][j] = 0
                elif zero_count == 1:
                    if grid[i][j] == 0:
                        p[i][j] = total_product % 12345
                    else:
                        p[i][j] = 0
                else:
                    p[i][j] = (total_product * pow(grid[i][j], 12343, 12345)) % 12345
        
        return p