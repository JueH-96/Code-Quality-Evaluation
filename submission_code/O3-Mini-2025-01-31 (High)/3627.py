from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # dist[i][j] is the earliest time you can arrive at (i, j)
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Heap holds tuples of (current_time, row, col)
        heap = [(0, 0, 0)]
        
        while heap:
            t, i, j = heapq.heappop(heap)
            
            # Skip if we already found a better time for cell (i,j)
            if t > dist[i][j]:
                continue
                
            # Return immediately if we reached the destination
            if i == n - 1 and j == m - 1:
                return t
            
            # Explore all four adjacent cells (up, down, left, right)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                # Check bounds
                if 0 <= ni < n and 0 <= nj < m:
                    # To move into cell (ni, nj), we need to wait until time is at least moveTime[ni][nj]
                    # Then it takes exactly 1 second to move.
                    new_time = max(t, moveTime[ni][nj]) + 1
                    # If we found a faster way to reach (ni, nj), update and push to the heap
                    if new_time < dist[ni][nj]:
                        dist[ni][nj] = new_time
                        heapq.heappush(heap, (new_time, ni, nj))
        
        # According to the problem statement, a path always exists.
        return -1