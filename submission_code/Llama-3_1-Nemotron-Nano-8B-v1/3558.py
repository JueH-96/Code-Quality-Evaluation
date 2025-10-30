from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        
        # Handle the case where start is the end cell
        if m == 1 and n == 1:
            return (health - grid[0][0]) >= 1
        
        # Initial health after starting cell
        initial_health = health - grid[0][0]
        if initial_health <= 0:
            return False
        
        # Directions: up, down, left, right
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize max_health matrix to track the maximum health at each cell
        max_health = [[-1 for _ in range(n)] for _ in range(m)]
        max_health[0][0] = initial_health
        
        # Queue for BFS, storing (i, j)
        q = deque([(0, 0)])
        
        while q:
            i, j = q.popleft()
            
            # Check if current cell is the destination
            if i == m - 1 and j == n - 1:
                return True
            
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    # Calculate new health after moving to (ni, nj)
                    new_health = max_health[i][j] - grid[ni][nj]
                    # Update if new_health is better and valid
                    if new_health >= 1 and new_health > max_health[ni][nj]:
                        max_health[ni][nj] = new_health
                        q.append((ni, nj))
        
        # If destination is not reachable
        return False