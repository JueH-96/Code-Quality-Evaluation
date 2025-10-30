class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        def check(row):
            return sum(row[i] == row[n - 1 - i] for i in range(n // 2))
        if m % 2 == 0:
            return sum(check(row) for row in grid) // 2
        else:
            mid = sum(check(row) for row in grid[:m // 2]) + sum(check(row) for row in grid[m // 2 + 1:])
            return min(mid, sum(check(row) for row in grid[m // 2:m // 2 + 1]) + sum(check(row) for row in grid[:m // 2]) + sum(check(row) for row in grid[m // 2 + 1:]))