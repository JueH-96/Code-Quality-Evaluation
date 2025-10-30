from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from heapq import heappush, heappop
        m, n = len(grid), len(grid[0])
        if m == 0 or n == 0:
            return False
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize distance matrix with infinity
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        
        # Priority queue: (distance, (i,j))
        heap = [(dist[0][0], (0, 0))]
        
        visited = set()
        
        while heap:
            dist_current, (i, j) = heappop(heap)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            
            # If reached the destination
            if i == m - 1 and j == n - 1:
                return dist_current <= health - 1
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_dist = dist_current + grid[ni][nj]
                    if new_dist < dist[ni][nj]:
                        dist[ni][nj] = new_dist
                        heappush(heap, (new_dist, (ni, nj)))
        
        return False