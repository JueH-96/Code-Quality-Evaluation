class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = set()
        
        def dfs(x, y, h):
            if h <= 0:
                return False
            if (x, y, h) in visited:
                return False
            visited.add((x, y, h))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= m or ny >= n or grid[nx][ny] == 1:
                    continue
                if dfs(nx, ny, h - 1):
                    return True
            return False
        
        return dfs(0, 0, health)