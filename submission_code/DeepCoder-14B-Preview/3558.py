from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        if m == 0:
            return False
        n = len(grid[0])
        if n == 0:
            return False
        
        # Calculate initial health after stepping into the starting cell
        start_health = health - grid[0][0]
        if start_health <= 0:
            return False
        
        # Visited matrix to keep track of the maximum health at each cell
        visited = [[-1 for _ in range(n)] for _ in range(m)]
        visited[0][0] = start_health
        
        # Initialize the queue with the starting position and health
        queue = deque()
        queue.append((0, 0, start_health))
        
        # Define the four possible movement directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            i, j, curr_health = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    # Calculate new health after moving to the new cell
                    new_health = curr_health - grid[ni][nj]
                    
                    # Check if the new cell is the destination
                    if ni == m - 1 and nj == n - 1:
                        if new_health >= 1:
                            return True
                    else:
                        # For non-destination cells, health must be positive after moving
                        if new_health > 0:
                            # Only proceed if this path offers better health than previously recorded
                            if visited[ni][nj] < new_health:
                                visited[ni][nj] = new_health
                                queue.append((ni, nj, new_health))
        
        # If queue is exhausted and destination not reached, return False
        return False