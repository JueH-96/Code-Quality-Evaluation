from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Directions: up, down, left, right
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        # Initialize earliest_times with infinity
        earliest_times = [[float('inf')] * m for _ in range(n)]
        earliest_times[0][0] = 0  # Starting point
        
        # Priority queue: (time, row, col)
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        
        while heap:
            current_time, row, col = heapq.heappop(heap)
            
            # If we've reached the destination, return the time
            if row == n - 1 and col == m - 1:
                return current_time
            
            # If current_time is greater than recorded time, skip
            if current_time > earliest_times[row][col]:
                continue
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                # Check if new position is within bounds
                if 0 <= nr < n and 0 <= nc < m:
                    # Start time to move to neighbor is max of current_time and moveTime[nr][nc]
                    start_time = max(current_time, moveTime[nr][nc])
                    # Arrival time is start_time + 1
                    arrival_time = start_time + 1
                    # If this arrival_time is better than recorded, update and push to heap
                    if arrival_time < earliest_times[nr][nc]:
                        earliest_times[nr][nc] = arrival_time
                        heapq.heappush(heap, (arrival_time, nr, nc))
        
        # If destination is not reachable, return -1
        return -1