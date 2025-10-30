class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        q = [(0, 0, health)]
        visited = set()
        visited.add((0, 0))
        while q:
            x, y, h = q.pop(0)
            if x == m - 1 and y == n - 1:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    nh = h - grid[nx][ny]
                    if nh > 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny, nh))
        return False