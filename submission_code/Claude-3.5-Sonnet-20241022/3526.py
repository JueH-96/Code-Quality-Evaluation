class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Count flips needed to make each row palindromic
        def count_row_flips():
            total = 0
            for i in range(m):
                for j in range(n//2):
                    if grid[i][j] != grid[i][n-1-j]:
                        total += 1
            return total
        
        # Count flips needed to make each column palindromic
        def count_col_flips():
            total = 0
            for j in range(n):
                for i in range(m//2):
                    if grid[i][j] != grid[m-1-i][j]:
                        total += 1
            return total
            
        # For rows, we need to count flips for non-matching pairs
        row_flips = count_row_flips()
        
        # For columns, we need to count flips for non-matching pairs
        col_flips = count_col_flips()
        
        # Return minimum of flips needed for rows or columns
        return min(row_flips, col_flips)