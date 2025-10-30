class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]

        def dfs(row, col, current_health):
            if row < 0 or row >= m or col < 0 or col >= n or current_health <= 0 or visited[row][col]:
                return False
            if row == m - 1 and col == n - 1:
                return True

            visited[row][col] = True
            next_health = current_health - grid[row][col]

            if (dfs(row + 1, col, next_health) or
                dfs(row - 1, col, next_health) or
                dfs(row, col + 1, next_health) or
                dfs(row, col - 1, next_health)):
                return True
            
            visited[row][col] = False # Backtrack for other paths
            return False

        return dfs(0, 0, health)