from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        if not grid or not grid[0]:
            return False
        
        m, n = len(grid), len(grid[0])
        # When stepping into the starting cell (0,0), subtract health if it's unsafe.
        start_health = health - grid[0][0]
        if start_health <= 0:
            return False
        
        # best[r][c] will keep track of the maximum health we've had reaching cell (r, c).
        best = [[-1] * n for _ in range(m)]
        best[0][0] = start_health
        
        dq = deque()
        dq.append((0, 0, start_health))
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while dq:
            r, c, cur_health = dq.popleft()
            # If we've reached the destination with positive health, return True.
            if r == m - 1 and c == n - 1:
                return True
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:  # check bounds
                    # Deduct health if stepping on an unsafe cell.
                    new_health = cur_health - grid[nr][nc]
                    # Continue only if new health is positive.
                    if new_health <= 0:
                        continue
                    # If we've reached (nr, nc) with a better health, update and add to queue.
                    if new_health > best[nr][nc]:
                        best[nr][nc] = new_health
                        dq.append((nr, nc, new_health))
        
        return False