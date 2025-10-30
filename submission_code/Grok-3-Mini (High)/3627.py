import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        pq = []
        heapq.heappush(pq, (0, 0, 0))  # arrival_time, row, col
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            curr_time, i, j = heapq.heappop(pq)
            if curr_time > dist[i][j]:
                continue  # Skip outdated entries
            
            # Check all four adjacent cells
            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0 <= ni < n and 0 <= nj < m:  # Within bounds
                    t_start = max(curr_time, moveTime[ni][nj])
                    arrival_at_nbr = t_start + 1
                    if arrival_at_nbr < dist[ni][nj]:
                        dist[ni][nj] = arrival_at_nbr
                        heapq.heappush(pq, (arrival_at_nbr, ni, nj))
        
        return int(dist[n-1][m-1])