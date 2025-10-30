import collections
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        
        # Initial health after entering the start cell
        init_health = health - grid[0][0]
        if init_health <= 0:
            return False
        
        # BFS setup
        queue = collections.deque()
        queue.append((0, 0, init_health))
        visited = set()
        visited.add((0, 0, init_health))
        
        # Directions for moving up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            ci, cj, ch = queue.popleft()
            
            # Check if reached the end cell
            if ci == m - 1 and cj == n - 1:
                return True
            
            # Explore adjacent cells
            for di, dj in directions:
                ni = ci + di
                nj = cj + dj
                if 0 <= ni < m and 0 <= nj < n:  # Within bounds
                    new_h = ch - grid[ni][nj]
                    if new_h > 0 and (ni, nj, new_h) not in visited:
                        queue.append((ni, nj, new_h))
                        visited.add((ni, nj, new_h))
        
        return False