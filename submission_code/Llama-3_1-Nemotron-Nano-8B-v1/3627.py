import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_time, i, j = heapq.heappop(heap)
            if i == n - 1 and j == m - 1:
                return current_time
            if current_time > dist[i][j]:
                continue
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m:
                    start_time = max(current_time, moveTime[ni][nj])
                    arrival_time = start_time + 1
                    if arrival_time < dist[ni][nj]:
                        dist[ni][nj] = arrival_time
                        heapq.heappush(heap, (arrival_time, ni, nj))
        return dist[n-1][m-1]