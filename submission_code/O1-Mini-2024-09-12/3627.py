from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Directions: up, down, left, right
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        # Initialize the minimum time grid with infinity
        min_time = [[float('inf')] * m for _ in range(n)]
        min_time[0][0] = 0
        
        # Priority queue: (current_time, x, y)
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        
        while heap:
            current_time, x, y = heapq.heappop(heap)
            
            # If we've reached the destination, return the time
            if x == n -1 and y == m -1:
                return current_time
            
            # If we have already found a better path to (x, y), skip
            if current_time > min_time[x][y]:
                continue
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check boundaries
                if 0 <= nx < n and 0 <= ny < m:
                    # Calculate the earliest time we can depart from current cell
                    depart_time = max(current_time, moveTime[x][y])
                    arrive_time = depart_time + 1
                    
                    # If arriving time is better, update and push to heap
                    if arrive_time < min_time[nx][ny]:
                        min_time[nx][ny] = arrive_time
                        heapq.heappush(heap, (arrive_time, nx, ny))
        
        # If the destination is unreachable, return -1 (based on problem constraints this shouldn't happen)
        return -1