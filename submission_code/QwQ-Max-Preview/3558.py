from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        max_allowed_damage = health - 1
        
        # Check if starting cell's damage already exceeds the allowed
        initial_damage = grid[0][0]
        if initial_damage > max_allowed_damage:
            return False
        
        # Directions: up, down, left, right
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize the queue with starting position and initial damage
        queue = deque()
        queue.append((0, 0, initial_damage))
        
        # min_damage[i][j] keeps the minimum damage required to reach (i,j)
        min_damage = [[float('inf')] * n for _ in range(m)]
        min_damage[0][0] = initial_damage
        
        while queue:
            i, j, current_damage = queue.popleft()
            
            # Check if we've reached the destination
            if i == m - 1 and j == n - 1:
                return True
            
            # Explore all four directions
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_damage = current_damage + grid[ni][nj]
                    # Check if new damage is within allowed limit and better than previous
                    if new_damage <= max_allowed_damage and new_damage < min_damage[ni][nj]:
                        min_damage[ni][nj] = new_damage
                        queue.append((ni, nj, new_damage))
        
        # If queue is empty and destination not reached, return False
        return False