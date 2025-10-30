class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def min_flips_to_palindrome(arr):
            n = len(arr)
            flips = 0
            for i in range(n // 2):
                if arr[i] != arr[n - i - 1]:
                    flips += 1
            return flips
        
        m, n = len(grid), len(grid[0])
        
        row_flips = sum(min_flips_to_palindrome(row) for row in grid)
        col_flips = sum(min_flips_to_palindrome([grid[i][j] for i in range(m)]) for j in range(n))
        
        return min(row_flips, col_flips)