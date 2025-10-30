class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        cost1 = 0
        for i in range(m):
            for j in range(n // 2):
                if grid[i][j] != grid[i][n-1-j]:
                    cost1 += 1
        cost2 = 0
        for j in range(n):
            for i in range(m // 2):
                if grid[i][j] != grid[m-1-i][j]:
                    cost2 += 1
        return min(cost1, cost2)