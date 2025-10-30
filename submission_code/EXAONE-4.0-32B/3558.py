from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        
        if m == 1 and n == 1:
            return health - grid[0][0] >= 1
        
        dp = [[-10**9] * n for _ in range(m)]
        q = deque()
        
        start_health = health - grid[0][0]
        if start_health < 1:
            return False
        dp[0][0] = start_health
        q.append((0, 0, start_health))
        
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while q:
            i, j, h = q.popleft()
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n:
                    new_health = h - grid[ni][nj]
                    if new_health < 1:
                        continue
                    if new_health > dp[ni][nj]:
                        dp[ni][nj] = new_health
                        q.append((ni, nj, new_health))
        
        return dp[m-1][n-1] >= 1