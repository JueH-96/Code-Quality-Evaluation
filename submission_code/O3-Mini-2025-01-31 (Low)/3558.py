from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        # Adjust initial health due to starting cell if it is unsafe.
        start_health = health - grid[0][0]
        if start_health < 1:
            return False
        
        # Directions for adjacent cells (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Use BFS: each state is (x, y, curr_health)
        queue = deque([(0, 0, start_health)])
        
        # We keep track of the best health we have arrived with at (i,j).
        visited = [[-1]*n for _ in range(m)]
        visited[0][0] = start_health
        
        while queue:
            i, j, curr_health = queue.popleft()
            # Check if reached target
            if i == m-1 and j == n-1:
                return True
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    # Calculate new health after stepping in (ni, nj)
                    new_health = curr_health - grid[ni][nj]
                    if new_health >= 1 and new_health > visited[ni][nj]:
                        visited[ni][nj] = new_health
                        queue.append((ni, nj, new_health))
        
        return False