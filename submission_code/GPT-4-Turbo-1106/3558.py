from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(x, y, health):
            # If we reach the destination with health greater than or equal to 1, return True
            if x == m - 1 and y == n - 1:
                return health >= 1

            # If we are on an unsafe cell, reduce health
            if grid[x][y] == 1:
                health -= 1

            # If health drops to 0 or below, return False
            if health <= 0:
                return False

            # Mark the cell as visited by setting it to 2
            grid[x][y] = 2

            # Explore all possible directions
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] != 2:
                    if dfs(new_x, new_y, health):
                        return True

            # Unmark the cell before backtracking
            grid[x][y] = 0
            return False

        return dfs(0, 0, health)