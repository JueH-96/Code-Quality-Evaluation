import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        from typing import List
        
        n = len(moveTime)
        m = len(moveTime[0])
        DIR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize arrival times with infinity
        arrival_times = [[float('inf')] * m for _ in range(n)]
        arrival_times[0][0] = 0
        
        # Priority queue: (arrival_time, i, j)
        heap = [(0, 0, 0)]
        
        while heap:
            t, i, j = heapq.heappop(heap)
            
            # If this room is the destination, return its arrival time
            if i == n - 1 and j == m - 1:
                return t
            
            # If this room has already been processed with a better time, skip
            if t > arrival_times[i][j]:
                continue
            
            # Explore all adjacent rooms
            for di, dj in DIR:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    # Time to start moving to the next room
                    start_time = max(t + 1, moveTime[ni][nj])
                    # Arrival time to the next room
                    arrival_time_next = start_time + 1
                    # If this arrival time is better, update and push to heap
                    if arrival_time_next < arrival_times[ni][nj]:
                        arrival_times[ni][nj] = arrival_time_next
                        heapq.heappush(heap, (arrival_time_next, ni, nj))
        
        # If the destination is reachable, the arrival time is in arrival_times[-1][-1]
        return arrival_times[-1][-1]