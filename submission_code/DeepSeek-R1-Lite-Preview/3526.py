class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        # Calculate flips needed for rows
        flips_rows = 0
        for row in grid:
            for j in range(n // 2):
                if row[j] != row[n - 1 - j]:
                    flips_rows += 1
        
        # Calculate flips needed for columns
        flips_columns = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    flips_columns += 1
        
        # Return the minimum of the two
        return min(flips_rows, flips_columns)