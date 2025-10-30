class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def min_flips_to_palindrome(line):
            return sum(a != b for a, b in zip(line, reversed(line))) // 2
        
        m, n = len(grid), len(grid[0])
        
        # Check if all rows or all columns are already palindromic
        if all(row == row[::-1] for row in grid) or all(col == col[::-1] for col in zip(*grid)):
            return 0
        
        # Calculate the minimum flips for rows and columns
        row_flips = sum(min_flips_to_palindrome(row) for row in grid)
        col_flips = sum(min_flips_to_palindrome(col) for col in zip(*grid))
        
        # Return the minimum of the two
        return min(row_flips, col_flips)