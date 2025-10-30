class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]

        def dfs(r, c, current_health):
            if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or current_health <= 0:
                return False

            visited[r][c] = True
            damage = grid[r][c]
            next_health = current_health - damage

            if r == rows - 1 and c == cols - 1:
                return next_health >= 1

            if dfs(r + 1, c, next_health) or \
               dfs(r - 1, c, next_health) or \
               dfs(r, c + 1, next_health) or \
               dfs(r, c - 1, next_health):
                return True

            visited[r][c] = False # Backtrack
            return False

        return dfs(0, 0, health)