from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        # Priority queue for Dijkstra's algorithm
        pq = [(0, 0, 0)]  # (current_time, x, y)
        # To keep track of the minimum time to reach each room
        min_time = [[float('inf')] * m for _ in range(n)]
        min_time[0][0] = 0
        
        # Directions for moving up, down, left, right
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while pq:
            current_time, x, y = heapq.heappop(pq)
            
            # If we reached the bottom-right corner
            if x == n - 1 and y == m - 1:
                return current_time
            
            # Explore adjacent rooms
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # Calculate the time to move to the adjacent room
                    next_time = max(current_time + 1, moveTime[nx][ny])
                    # If we found a better time to reach (nx, ny)
                    if next_time < min_time[nx][ny]:
                        min_time[nx][ny] = next_time
                        heapq.heappush(pq, (next_time, nx, ny))
        
        return -1  # In case there's no way to reach (n-1, m-1), though the problem guarantees a path