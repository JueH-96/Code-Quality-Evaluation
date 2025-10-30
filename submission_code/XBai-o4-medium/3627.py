import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        INF = float('inf')
        dist = [[INF] * m for _ in range(n)]
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
            for di, dj in directions:
                ni = i + di
                nj = j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    new_time = max(current_time, moveTime[ni][nj]) + 1
                    if new_time < dist[ni][nj]:
                        dist[ni][nj] = new_time
                        heapq.heappush(heap, (new_time, ni, nj))
        return -1