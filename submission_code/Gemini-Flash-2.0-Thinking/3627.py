import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])

        min_arrival_time = [[float('inf')] * m for _ in range(n)]
        min_arrival_time[0][0] = 0

        pq = [(0, 0, 0)]  # (arrival_time, row, col)

        while pq:
            current_arrival_time, r, c = heapq.heappop(pq)

            if current_arrival_time > min_arrival_time[r][c]:
                continue

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                if 0 <= nr < n and 0 <= nc < m:
                    start_move_time = max(current_arrival_time, moveTime[nr][nc])
                    arrival_time = start_move_time + 1

                    if arrival_time < min_arrival_time[nr][nc]:
                        min_arrival_time[nr][nc] = arrival_time
                        heapq.heappush(pq, (arrival_time, nr, nc))

        return min_arrival_time[n - 1][m - 1]