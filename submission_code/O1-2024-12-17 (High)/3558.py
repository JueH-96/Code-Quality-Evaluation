class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque

        m, n = len(grid), len(grid[0])

        # The amount of health used so far, starting at (0, 0).
        start_used = grid[0][0]
        # If we already exceed or match the health upon entry, we can't proceed (no health left).
        if start_used >= health:
            return False

        # visited[x][y][used] indicates whether we have visited (x, y) having expended "used" health.
        # We only need to track used < health, because used == health means no health remaining.
        visited = [[[False] * health for _ in range(n)] for _ in range(m)]
        visited[0][0][start_used] = True

        # Standard BFS queue, storing (row, column, used_health).
        queue = deque([(0, 0, start_used)])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while queue:
            x, y, used = queue.popleft()

            # If we've reached the bottom-right corner with used < health, we have health >= 1 left.
            if x == m - 1 and y == n - 1:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_used = used + grid[nx][ny]
                    # Only proceed if we have not exhausted health and haven't visited this state yet.
                    if new_used < health and not visited[nx][ny][new_used]:
                        visited[nx][ny][new_used] = True
                        queue.append((nx, ny, new_used))

        # If all possible states are exhausted without reaching the destination safely, return False.
        return False