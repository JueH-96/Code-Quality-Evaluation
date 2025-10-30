from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        min_time = [[float('inf')] * m for _ in range(n)]
        min_time[0][0] = 0
        
        # Priority queue to process cells in increasing order of their earliest possible visit time
        pq = [(0, 0, 0)]  # (current_time, x, y)
        heapq.heapify(pq)
        
        while pq:
            current_time, x, y = heapq.heappop(pq)
            
            # If we reached the bottom-right corner
            if x == n - 1 and y == m - 1:
                return current_time
            
            # Explore all possible moves
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    # Calculate the earliest time we can move into the next room
                    next_time = max(current_time + 1, moveTime[nx][ny])
                    if next_time < min_time[nx][ny]:
                        min_time[nx][ny] = next_time
                        heapq.heappush(pq, (next_time, nx, ny))
        
        return min_time[n-1][m-1]