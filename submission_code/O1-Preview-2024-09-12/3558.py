from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        maxHealth = [[-1]*n for _ in range(m)]
        queue = deque()
        queue.append( (0, 0, health) )
        maxHealth[0][0] = health
        
        while queue:
            i, j, h = queue.popleft()
            
            if i == m - 1 and j == n - 1:
                return True  # Reached the goal with positive health
            
            for dir_i, dir_j in [ (0, 1), (1, 0), (-1, 0), (0, -1) ]:
                ni, nj = i + dir_i, j + dir_j
                if 0 <= ni < m and 0 <= nj < n:
                    nh = h
                    if grid[ni][nj] == 1:
                        nh -=1
                    if nh >=1:
                        if maxHealth[ni][nj] < nh:
                            maxHealth[ni][nj] = nh
                            queue.append( (ni, nj, nh) )
        
        return False