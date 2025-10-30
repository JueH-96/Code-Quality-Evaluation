import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        pq = [(0, 0, 0)]  # (leave_time, r, c)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while pq:
            leave_time, r, c = heapq.heappop(pq)

            if leave_time > dist[r][c]:
                continue
            
            if r == n - 1 and c == m - 1:
                return leave_time + 1

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    arrival_time = leave_time + 1
                    start_move_time = moveTime[nr][nc]
                    next_leave_time = max(arrival_time, start_move_time)
                    if next_leave_time < dist[nr][nc]:
                        dist[nr][nc] = next_leave_time
                        heapq.heappush(pq, (next_leave_time, nr, nc))
        return -1