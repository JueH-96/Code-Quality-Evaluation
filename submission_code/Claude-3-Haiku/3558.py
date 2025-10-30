from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        queue = deque([(0, 0, health)])
        visited = set()

        while queue:
            x, y, h = queue.popleft()
            if (x, y) == (m-1, n-1):
                return h >= 1
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    nh = h - grid[nx][ny]
                    if nh >= 0:
                        queue.append((nx, ny, nh))

        return False