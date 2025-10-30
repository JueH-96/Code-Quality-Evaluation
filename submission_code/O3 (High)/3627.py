from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        """
        Dijkstra on a grid where every edge costs 1 second plus a possible waiting
        time that depends on the destination cell.

        From a cell reached at `t`, going to a neighbour (r,c) is allowed only
        if you start moving no earlier than  `moveTime[r][c]`.  Hence

            earliest_start = max(t, moveTime[r][c])
            arrival_time   = earliest_start + 1

        We keep the smallest arrival‐time for every cell with a priority queue.
        """
        n, m = len(moveTime), len(moveTime[0])
        INF = 10**20
        dist = [[INF] * m for _ in range(n)]
        dist[0][0] = 0
        pq = [(0, 0, 0)]                        # (current time, row, col)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while pq:
            t, r, c = heapq.heappop(pq)
            if (r, c) == (n - 1, m - 1):        # reached target
                return t
            if t != dist[r][c]:                 # outdated entry
                continue

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    start = max(t, moveTime[nr][nc])
                    arrive = start + 1
                    if arrive < dist[nr][nc]:
                        dist[nr][nc] = arrive
                        heapq.heappush(pq, (arrive, nr, nc))

        return -1                                # grid is always reachable per constraints