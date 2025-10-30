from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        # Initialize a DP table to store the maximum health at each cell
        dp = [[-1 for _ in range(n)] for _ in range(m)]
        dp[0][0] = health - grid[0][0]
        if dp[0][0] < 0:
            return False
        # Directions for moving up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque()
        queue.append((0, 0))
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_health = dp[x][y] - grid[nx][ny]
                    if new_health > dp[nx][ny] and new_health >= 0:
                        dp[nx][ny] = new_health
                        queue.append((nx, ny))
        return dp[m-1][n-1] >= 0