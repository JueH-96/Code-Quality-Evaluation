from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        # dijkstra: state: (time, r, c)
        # Define directions: up, down, left, right.
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        # distance matrix
        dist = [[float('inf')] * m for _ in range(n)]
        # Starting at (0, 0) at time 0, note that we can always start there if t >= moveTime[0][0]?
        # Actually, since moveTime[0][0] indicates when you can "start moving to that room", and you're already there,
        # we assume you can start moving immediately regardless.
        dist[0][0] = 0
        
        # Use min heap
        heap = [(0, 0, 0)]  # (time, row, col)
        
        while heap:
            t, r, c = heapq.heappop(heap)
            if t > dist[r][c]:
                continue
            # if reached target, we can return because it's the minimal time.
            if r == n - 1 and c == m - 1:
                return t
            # Process neighbors.
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    # To move to neighbor (nr, nc), you must wait until time >= moveTime[nr][nc]
                    # So departure time = max(t, moveTime[nr][nc]), then plus one second for the move.
                    new_time = max(t, moveTime[nr][nc]) + 1
                    if new_time < dist[nr][nc]:
                        dist[nr][nc] = new_time
                        heapq.heappush(heap, (new_time, nr, nc))
                        
        # If not reachable, though by constraints it's always reachable.
        return -1