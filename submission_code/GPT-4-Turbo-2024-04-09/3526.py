class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def min_flips_to_palindrome(line):
            n = len(line)
            return sum(line[i] != line[n - 1 - i] for i in range(n // 2))
        
        m, n = len(grid), len(grid[0])
        
        # Calculate flips required to make all rows palindromic
        row_flips = sum(min_flips_to_palindrome(grid[i]) for i in range(m))
        
        # Calculate flips required to make all columns palindromic
        col_flips = sum(min_flips_to_palindrome([grid[i][j] for i in range(m)]) for j in range(n))
        
        # Return the minimum of the two
        return min(row_flips, col_flips)