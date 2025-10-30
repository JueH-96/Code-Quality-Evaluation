from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        queue = deque([(0, 0, health)])
        visited = set([(0, 0)])

        while queue:
            x, y, current_health = queue.popleft()

            if x == m - 1 and y == n - 1:
                return current_health > 0

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy

                if 0 <= new_x < m and 0 <= new_y < n and (new_x, new_y) not in visited:
                    new_health = current_health - grid[new_x][new_y]

                    if new_health > 0:
                        queue.append((new_x, new_y, new_health))
                        visited.add((new_x, new_y))

        return False