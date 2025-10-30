from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = deque([(0, 0, health, [])])
        visited = set()

        while queue:
            x, y, health, path = queue.popleft()
            if (x, y, health) in visited:
                continue
            visited.add((x, y, health))

            if grid[x][y] == 1:
                health -= 1
                if health == 0:
                    continue

            if (x, y) == (m - 1, n - 1):
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny, health - 1) not in visited:
                    queue.append((nx, ny, health - 1, path + [(dx, dy)]))

        return False