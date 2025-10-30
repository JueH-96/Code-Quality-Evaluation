from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        # We need the total number of unsafe cells (grid=1) along path <= health-1
        max_unsafe = health - 1
        
        # Directions: up, down, left, right
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        # dist[i][j] = minimum number of unsafe cells to reach (i,j)
        dist = [[float('inf')] * n for _ in range(m)]
        
        # Min-heap for Dijkstra: (unsafe_count, i, j)
        heap = []
        start_cost = grid[0][0]
        if start_cost > max_unsafe:
            return False
        dist[0][0] = start_cost
        heapq.heappush(heap, (start_cost, 0, 0))
        
        while heap:
            cost, i, j = heapq.heappop(heap)
            # If this entry is stale, skip
            if cost > dist[i][j]:
                continue
            # If we've reached target, we can stop early
            if i == m-1 and j == n-1:
                return cost <= max_unsafe
            # Explore neighbors
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = cost + grid[ni][nj]
                    if new_cost < dist[ni][nj] and new_cost <= max_unsafe:
                        dist[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        
        # Check if we reached the end
        return dist[m-1][n-1] <= max_unsafe