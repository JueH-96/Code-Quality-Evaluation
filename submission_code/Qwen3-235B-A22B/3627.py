import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        pq = []
        heapq.heappush(pq, (0, 0, 0))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            current_time, x, y = heapq.heappop(pq)
            if x == n - 1 and y == m - 1:
                return current_time
            if current_time > dist[x][y]:
                continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m:
                    next_time = max(current_time, moveTime[nx][ny]) + 1
                    if next_time < dist[nx][ny]:
                        dist[nx][ny] = next_time
                        heapq.heappush(pq, (next_time, nx, ny))
        return -1