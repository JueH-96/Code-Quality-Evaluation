import heapq
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        
        start_health = health - grid[0][0]
        if start_health <= 0:
            return False
        
        best = [[-1 for _ in range(n)] for _ in range(m)]
        best[0][0] = start_health
        heap = []
        heapq.heappush(heap, (-start_health, 0, 0))
        
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_neg, i, j = heapq.heappop(heap)
            current = -current_neg
            
            if i == m - 1 and j == n - 1:
                if current >= 1:
                    return True
                continue
            
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_health = current - grid[ni][nj]
                    if new_health > 0 and new_health > best[ni][nj]:
                        best[ni][nj] = new_health
                        heapq.heappush(heap, (-new_health, ni, nj))
        
        return False