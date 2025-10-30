from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        if m == 0 or n == 0:
            return False
        
        # Directions: up, down, left, right
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        # Use a queue for BFS: (i, j, current_health)
        queue = deque()
        initial_health = health - grid[0][0]
        if initial_health <= 0:
            return False
        queue.append((0, 0, initial_health))
        
        # Visited dictionary to store max health at each cell
        visited = [[-1 for _ in range(n)] for _ in range(m)]
        visited[0][0] = initial_health
        
        while queue:
            i, j, h = queue.popleft()
            
            # If reached the end
            if i == m-1 and j == n-1:
                return True
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    nh = h - grid[ni][nj]
                    if nh > 0 and nh > visited[ni][nj]:
                        visited[ni][nj] = nh
                        queue.append((ni, nj, nh))
        
        return False