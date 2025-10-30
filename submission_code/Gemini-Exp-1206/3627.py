import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        pq = [(0, 0, 0)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while pq:
            d, r, c = heapq.heappop(pq)

            if d > dist[r][c]:
                continue

            if r == n - 1 and c == m - 1:
                return d

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    wait_time = max(0, moveTime[nr][nc] - d - 1)
                    if (wait_time % 2) != 0:
                        wait_time += 1
                    nd = d + 1 + wait_time
                    if nd < dist[nr][nc]:
                        dist[nr][nc] = nd
                        heapq.heappush(pq, (nd, nr, nc))

        return -1