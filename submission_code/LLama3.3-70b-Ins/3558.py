from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        memo = {}

        def dfs(i: int, j: int, health: int) -> bool:
            if (i, j, health) in memo:
                return memo[(i, j, health)]
            if i < 0 or i >= m or j < 0 or j >= n or health <= 0:
                return False
            if i == m - 1 and j == n - 1:
                return health > 0
            health -= grid[i][j]
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                if dfs(i + dx, j + dy, health):
                    memo[(i, j, health + grid[i][j])] = True
                    return True
            memo[(i, j, health + grid[i][j])] = False
            return False

        return dfs(0, 0, health)