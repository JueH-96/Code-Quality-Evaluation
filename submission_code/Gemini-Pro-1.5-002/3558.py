class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        q = [(0, 0, health)]
        visited = set([(0, 0, health)])
        
        while q:
            r, c, h = q.pop(0)
            
            if r == m - 1 and c == n - 1:
                return h >= 1
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < m and 0 <= nc < n:
                    nh = h - grid[nr][nc]
                    if nh > 0 and (nr, nc, nh) not in visited:
                        visited.add((nr, nc, nh))
                        q.append((nr, nc, nh))
                        
        return False