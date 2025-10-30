class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        PA = [[0] * n for _ in range(m)]
        PB = [[0] * n for _ in range(m)]
        count = 0
        
        for i in range(m):
            for j in range(n):
                a_val = 1 if grid[i][j] == 'X' else 0
                b_val = 1 if grid[i][j] == 'Y' else 0
                
                topA = PA[i-1][j] if i-1 >= 0 else 0
                leftA = PA[i][j-1] if j-1 >= 0 else 0
                diagA = PA[i-1][j-1] if (i-1 >= 0 and j-1 >= 0) else 0
                PA[i][j] = a_val + topA + leftA - diagA
                
                topB = PB[i-1][j] if i-1 >= 0 else 0
                leftB = PB[i][j-1] if j-1 >= 0 else 0
                diagB = PB[i-1][j-1] if (i-1 >= 0 and j-1 >= 0) else 0
                PB[i][j] = b_val + topB + leftB - diagB
                
                if PA[i][j] == PB[i][j] and PA[i][j] > 0:
                    count += 1
                    
        return count