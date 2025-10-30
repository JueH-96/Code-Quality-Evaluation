class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        flip_count = 0
        
        # Flip rows
        for i in range(m):
            if grid[i] != grid[i][::-1]:
                grid[i] = grid[i][::-1]
                flip_count += 1
        
        # Flip columns
        for j in range(n):
            column = [row[j] for row in grid]
            if column != column[::-1]:
                for i in range(m):
                    grid[i][j] = 1 - grid[i][j]
                flip_count += 1
        
        return flip_count