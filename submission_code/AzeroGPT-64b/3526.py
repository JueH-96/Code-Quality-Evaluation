class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        # Checking if grid is already a palindrome in rows or columns
        if all(row == row[::-1] for row in grid) or all(col == col[::-1] for col in zip(*grid)):
            return 0

        # Calculating the cost to make rows palindromic
        row_cost = sum(min(row[i] ^ row[~i] for i in range(n // 2))
                       for row in grid)

        # Calculating the cost to make columns palindromic (ignoring row 0 for optimization)
        col_cost = float('inf')
        if m > 1:
            col_cost = sum(min(row[i] ^ row[~i] for row in grid[1:])
                           for i in range(n // 2))
            if n % 2 == 1:
                col_cost += min(grid[0][n // 2], sum(row[n // 2] for row in grid[1:]))

        return min(row_cost, col_cost)