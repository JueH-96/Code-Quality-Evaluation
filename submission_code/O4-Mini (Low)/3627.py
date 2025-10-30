from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        # Directions: up, down, left, right
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        # Distance array initialized to infinity
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Min-heap for Dijkstra: (current_time, x, y)
        heap = [(0, 0, 0)]
        
        while heap:
            t, x, y = heapq.heappop(heap)
            if t > dist[x][y]:
                continue
            # If we've reached bottom-right, we can return immediately
            if x == n-1 and y == m-1:
                return t
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # We can start moving when current time >= moveTime[nx][ny].
                    # If current time is less, we wait until moveTime[nx][ny].
                    start_move = max(t, moveTime[nx][ny])
                    arrival = start_move + 1
                    if arrival < dist[nx][ny]:
                        dist[nx][ny] = arrival
                        heapq.heappush(heap, (arrival, nx, ny))
        
        # In a valid dungeon there should always be a path
        return -1