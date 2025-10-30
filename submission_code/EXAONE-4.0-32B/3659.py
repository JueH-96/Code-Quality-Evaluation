mod = 10**9 + 7

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        
        if m == 1 and n == 1:
            return 1 if grid[0][0] == k else 0
        
        prev_row = [[0] * 16 for _ in range(n)]
        start_val = grid[0][0]
        prev_row[0][start_val] = 1
        
        for j in range(1, n):
            v = grid[0][j]
            for x in range(16):
                prev_row[j][x] = prev_row[j-1][x ^ v]
        
        for i in range(1, m):
            curr_row = [[0] * 16 for _ in range(n)]
            v0 = grid[i][0]
            for x in range(16):
                curr_row[0][x] = prev_row[0][x ^ v0]
            
            for j in range(1, n):
                v = grid[i][j]
                for x in range(16):
                    total = prev_row[j][x ^ v] + curr_row[j-1][x ^ v]
                    curr_row[j][x] = total % mod
            prev_row = curr_row
        
        return prev_row[n-1][k] % mod