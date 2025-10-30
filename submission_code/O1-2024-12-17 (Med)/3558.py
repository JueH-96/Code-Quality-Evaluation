class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        # We want to see if there's a path that uses at most (health - 1) unsafe cells
        # because we must end with at least 1 health point remaining.

        # Distance array where dist[r][c] = min number of '1' cells encountered up to (r, c).
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = grid[0][0]

        # Use a 0-1 BFS approach since moving onto a '0' cell costs 0 additional health
        # and moving onto a '1' cell costs 1 additional health.
        dq = deque()
        dq.append((0, 0))

        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        
        while dq:
            r, c = dq.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    cost = dist[r][c] + grid[nr][nc]
                    # Only proceed if this path is better and doesn't exceed (health - 1)
                    if cost < dist[nr][nc] and cost <= health - 1:
                        dist[nr][nc] = cost
                        # If this move cost 1 (grid[nr][nc] == 1), push to back; else front
                        if grid[nr][nc] == 1:
                            dq.append((nr, nc))
                        else:
                            dq.appendleft((nr, nc))

        return dist[m-1][n-1] <= health - 1