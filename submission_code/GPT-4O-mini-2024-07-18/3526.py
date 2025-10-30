class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        def min_flips_for_palindrome(arr):
            count = 0
            half = (len(arr) + 1) // 2
            for i in range(half):
                count += min(arr[i], 1 - arr[i]) + min(arr[-(i + 1)], 1 - arr[-(i + 1)])
            return count
        
        # Calculate flips needed for all rows to be palindromic
        row_flips = sum(min_flips_for_palindrome(grid[i]) for i in range(m))
        
        # Calculate flips needed for all columns to be palindromic
        col_flips = 0
        for j in range(n):
            col = [grid[i][j] for i in range(m)]
            col_flips += min_flips_for_palindrome(col)
        
        return min(row_flips, col_flips)