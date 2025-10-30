from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        if m == 0 or n == 0:
            return False  # Though constraints say m*n >=2
        
        # Initialize distance matrix with infinity
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        
        dq = deque()
        dq.append((0, 0))
        
        # Directions: up, down, left, right
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while dq:
            i, j = dq.popleft()
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n:
                    new_dist = dist[i][j] + grid[ni][nj]
                    if new_dist < dist[ni][nj]:
                        dist[ni][nj] = new_dist
                        if grid[ni][nj] == 0:
                            dq.appendleft((ni, nj))
                        else:
                            dq.append((ni, nj))
        
        # Check if end is reachable and health is sufficient
        if dist[m-1][n-1] == float('inf'):
            return False
        return health >= dist[m-1][n-1] + 1