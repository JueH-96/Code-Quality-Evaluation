from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        INF = float('inf')
        distances = [[INF] * m for _ in range(n)]
        distances[0][0] = 0
        heap = []
        heapq.heappush(heap, (0, 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            current_time, i, j = heapq.heappop(heap)
            if i == n - 1 and j == m - 1:
                return current_time
            if current_time > distances[i][j]:
                continue
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m:
                    new_time = max(current_time, moveTime[ni][nj]) + 1
                    if new_time < distances[ni][nj]:
                        distances[ni][nj] = new_time
                        heapq.heappush(heap, (new_time, ni, nj))
        return distances[n-1][m-1] if distances[n-1][m-1] != INF else -1