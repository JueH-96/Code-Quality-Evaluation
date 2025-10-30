from typing import List
import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])
        start_cost = grid[0][0]
        max_allowed = health - 1
        
        if start_cost > max_allowed:
            return False
        
        # Edge case: single cell
        if m == 1 and n == 1:
            return True
        
        visited = [[float('inf')] * n for _ in range(m)]
        visited[0][0] = start_cost
        
        heap = []
        heapq.heappush(heap, (start_cost, 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            curr_cost, i, j = heapq.heappop(heap)
            
            # Check if reached the end cell
            if i == m - 1 and j == n - 1:
                return True
            
            # Skip if a better path already found
            if curr_cost > visited[i][j]:
                continue
            
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < m and 0 <= nj < n:
                    new_cost = curr_cost + grid[ni][nj]
                    
                    if new_cost <= max_allowed and new_cost < visited[ni][nj]:
                        visited[ni][nj] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        
        return False