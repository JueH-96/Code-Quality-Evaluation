import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        pq = []
        heapq.heappush(pq, (0, 0, 0))
        
        while pq:
            time, i, j = heapq.heappop(pq)
            if i == n-1 and j == m-1:
                return time
            if time != dist[i][j]:
                continue
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    new_time = max(time, moveTime[ni][nj]) + 1
                    if new_time < dist[ni][nj]:
                        dist[ni][nj] = new_time
                        heapq.heappush(pq, (new_time, ni, nj))
        return dist[n-1][m-1]