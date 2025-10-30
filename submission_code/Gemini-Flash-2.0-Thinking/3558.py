from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])

        q = deque([(0, 0, health)])
        visited = set()

        while q:
            row, col, current_health = q.popleft()

            if (row, col, current_health) in visited:
                continue
            visited.add((row, col, current_health))

            if row == m - 1 and col == n - 1 and current_health >= 1:
                return True

            # Explore neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc

                if 0 <= new_row < m and 0 <= new_col < n:
                    new_health = current_health - grid[new_row][new_col]
                    if new_health > 0:
                        q.append((new_row, new_col, new_health))

        return False