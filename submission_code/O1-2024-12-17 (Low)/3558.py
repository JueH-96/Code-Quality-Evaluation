class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        
        # Adjust initial health based on the starting cell
        start_health = health - grid[0][0]
        if start_health <= 0:
            return False
        
        # visited[r][c] will store the maximum health we've had at (r, c) so far
        visited = [[-1] * n for _ in range(m)]
        visited[0][0] = start_health
        
        queue = deque()
        queue.append((0, 0, start_health))
        
        while queue:
            r, c, h = queue.popleft()
            
            # Check if we've reached the bottom-right cell with health >= 1
            if r == m - 1 and c == n - 1 and h >= 1:
                return True
            
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    # Decrease health if the next cell is unsafe
                    new_health = h - grid[nr][nc]
                    # Proceed only if we still have positive health and it's better than visited record
                    if new_health > visited[nr][nc] and new_health > 0:
                        visited[nr][nc] = new_health
                        queue.append((nr, nc, new_health))
        
        return False