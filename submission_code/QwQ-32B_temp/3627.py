import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0]) if n else 0
        INF = float('inf')
        dist = [[INF] * m for _ in range(n)]
        dist[0][0] = 0
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            t, i, j = heapq.heappop(heap)
            if i == n - 1 and j == m - 1:
                return t
            if t > dist[i][j]:
                continue
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    start_time = max(t, moveTime[ni][nj])
                    arrival = start_time + 1
                    if arrival < dist[ni][nj]:
                        dist[ni][nj] = arrival
                        heapq.heappush(heap, (arrival, ni, nj))
        return dist[n-1][m-1]