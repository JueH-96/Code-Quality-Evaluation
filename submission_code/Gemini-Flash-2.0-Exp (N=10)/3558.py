class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        if health <= 0:
            return False

        q = [(0, 0, health)]
        visited = set()

        while q:
            r, c, current_health = q.pop(0)

            if (r, c) == (m - 1, n - 1):
                return True

            if (r, c) in visited:
                continue
            visited.add((r,c))

            
            if grid[r][c] == 1:
                current_health -= 1
            
            if current_health <= 0:
                continue
            
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    q.append((nr, nc, current_health))
        
        return False