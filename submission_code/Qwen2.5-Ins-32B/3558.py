from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        
        # Initialize the queue with the starting point (0, 0)
        queue = deque([(0, 0, health)])
        visited = set()
        
        while queue:
            x, y, h = queue.popleft()
            
            # If the current cell is unsafe, decrease health
            if grid[x][y] == 1:
                h -= 1
            
            # If health becomes 0 or negative, continue to the next iteration
            if h <= 0:
                continue
            
            # If we reach the bottom-right corner with positive health, return True
            if x == m - 1 and y == n - 1:
                return True
            
            # Mark the current cell as visited
            visited.add((x, y))
            
            # Explore all four possible directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check if the new position is within bounds and not visited
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    queue.append((nx, ny, h))
        
        return False