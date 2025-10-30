from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        # dist[i][j] = minimum time to reach room (i, j)
        INF = 10**30
        dist = [[INF] * m for _ in range(n)]
        dist[0][0] = 0
        
        # Min-heap for Dijkstra: (current_time, i, j)
        pq = [(0, 0, 0)]
        # Four possible moves: up, down, left, right
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while pq:
            t, i, j = heapq.heappop(pq)
            # If we've already found a better way here, skip
            if t > dist[i][j]:
                continue
            # Early exit if we reached the target
            if i == n-1 and j == m-1:
                return t
            
            # Try moving to each neighbor
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    # You can only start moving into (ni, nj) at time >= moveTime[ni][nj],
                    # and also not before your current time t. Moving takes 1 second.
                    departure = max(t, moveTime[ni][nj])
                    arrival = departure + 1
                    if arrival < dist[ni][nj]:
                        dist[ni][nj] = arrival
                        heapq.heappush(pq, (arrival, ni, nj))
        
        # In case the target is unreachable (shouldn't happen under given constraints)
        return dist[n-1][m-1]