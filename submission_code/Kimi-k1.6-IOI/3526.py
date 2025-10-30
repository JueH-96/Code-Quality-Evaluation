class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        row_cost = 0
        
        # Calculate cost for making all rows palindromic
        for row in grid:
            row_len = len(row)
            for i in range(row_len // 2):
                if row[i] != row[row_len - 1 - i]:
                    row_cost += 1
        
        col_cost = 0
        # Calculate cost for making all columns palindromic
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m - 1 - i][j]:
                    col_cost += 1
        
        return min(row_cost, col_cost)