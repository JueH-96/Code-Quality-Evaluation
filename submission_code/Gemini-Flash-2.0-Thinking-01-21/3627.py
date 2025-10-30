from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        arrival_time = [[float('inf')] * m for _ in range(n)]
        arrival_time[0][0] = 0
        queue = [(0, 0)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            r, c = queue.pop(0)
            current_time = arrival_time[r][c]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m:
                    start_move_time = max(current_time, moveTime[nr][nc])
                    next_arrival_time = start_move_time + 1
                    if next_arrival_time < arrival_time[nr][nc]:
                        arrival_time[nr][nc] = next_arrival_time
                        queue.append((nr, nc))

        return arrival_time[n - 1][m - 1]