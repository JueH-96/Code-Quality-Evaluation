from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Distance array to keep track of the earliest time we can reach each cell
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Min-heap for Dijkstra-like approach: (time, row, col)
        min_heap = [(0, 0, 0)]
        
        # Directions for movement (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while min_heap:
            cur_time, r, c = heapq.heappop(min_heap)
            
            # If we've already found a better time for this cell, skip
            if cur_time > dist[r][c]:
                continue
            
            # If this is the bottom-right cell, return the time
            if r == n - 1 and c == m - 1:
                return cur_time
            
            # Check all valid neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    # We can move into (nr, nc) only after waiting until time >= moveTime[nr][nc]
                    # Then add 1 second of travel
                    new_time = max(cur_time, moveTime[nr][nc]) + 1
                    if new_time < dist[nr][nc]:
                        dist[nr][nc] = new_time
                        heapq.heappush(min_heap, (new_time, nr, nc))
        
        # If somehow we can't reach (n-1, m-1), though problem constraints suggest we always can
        return dist[n-1][m-1]