import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        # dist[i][j] will hold the minimum time to reach cell (i, j)
        INF = float('inf')
        dist = [[INF] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Min-heap for Dijkstra: (current_time, i, j)
        heap = [(0, 0, 0)]
        # Four possible directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while heap:
            cur_t, i, j = heapq.heappop(heap)
            # If we've already found a better way to (i, j), skip
            if cur_t > dist[i][j]:
                continue
            # If we've reached the target cell, return the time
            if i == n - 1 and j == m - 1:
                return cur_t
            # Explore neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    # We can only start moving into (ni,nj) at time >= moveTime[ni][nj]
                    # and the move itself takes 1 second
                    next_t = max(cur_t, moveTime[ni][nj]) + 1
                    if next_t < dist[ni][nj]:
                        dist[ni][nj] = next_t
                        heapq.heappush(heap, (next_t, ni, nj))
        
        # In a grid without walls it's always reachable, but we return this for completeness
        return dist[n-1][m-1]