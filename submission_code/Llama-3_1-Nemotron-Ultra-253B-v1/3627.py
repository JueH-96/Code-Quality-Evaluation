from heapq import heappush, heappop
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        arrival = [[float('inf')] * m for _ in range(n)]
        arrival[0][0] = 0
        heap = []
        heappush(heap, (0, 0, 0))
        
        while heap:
            time, i, j = heappop(heap)
            if i == n - 1 and j == m - 1:
                return time
            if time > arrival[i][j]:
                continue
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    new_time = max(time, moveTime[ni][nj]) + 1
                    if new_time < arrival[ni][nj]:
                        arrival[ni][nj] = new_time
                        heappush(heap, (new_time, ni, nj))
        return arrival[n-1][m-1]