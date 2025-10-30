class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        if grid[0][0] == 1:
            health -= 1
        if health <= 0:
            return False

        # Directions for moving up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue to always expand the state with the most health first
        from heapq import heappop, heappush
        pq = [(-health, 0, 0)]

        # Set to keep track of visited cells
        visited = set((0, 0))

        while pq:
            current_health, x, y = heappop(pq)
            current_health = -current_health

            # If we reach the bottom-right corner with health > 0, return True
            if x == m - 1 and y == n - 1:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    new_health = current_health - grid[nx][ny]
                    if new_health > 0:
                        heappush(pq, (-new_health, nx, ny))
                        visited.add((nx, ny))

        return False