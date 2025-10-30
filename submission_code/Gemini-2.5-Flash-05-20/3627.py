import heapq
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])

        # min_arrival_time[r][c] stores the minimum time to arrive at room (r, c)
        min_arrival_time = [[float('inf')] * m for _ in range(n)]

        # Priority queue: (current_time, row, col)
        # We start at (0, 0) at time t = 0
        pq = [(0, 0, 0)]
        min_arrival_time[0][0] = 0

        # Directions for adjacent rooms (up, down, left, right)
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        while pq:
            current_time, r, c = heapq.heappop(pq)

            # If we've already found a faster way to this room, skip
            # This check is crucial for performance in Dijkstra's when nodes can be pushed multiple times
            if current_time > min_arrival_time[r][c]:
                continue

            # If we reached the destination (bottom-right room)
            if r == n - 1 and c == m - 1:
                return current_time

            # Explore neighbors
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                # Check if the neighbor is within grid boundaries
                if 0 <= nr < n and 0 <= nc < m:
                    # Calculate the time to start moving from (r, c) to (nr, nc)
                    # This time depends on:
                    # 1. The time we arrived at the current room (r, c)
                    # 2. The minimum time (moveTime[nr][nc]) when a move TO (nr, nc) can start.
                    time_to_start_move = max(current_time, moveTime[nr][nc])

                    # The time to arrive at (nr, nc) is 1 second after starting the move
                    next_arrival_time = time_to_start_move + 1

                    # If this calculated path is shorter than a previously recorded path to (nr, nc)
                    if next_arrival_time < min_arrival_time[nr][nc]:
                        min_arrival_time[nr][nc] = next_arrival_time
                        heapq.heappush(pq, (next_arrival_time, nr, nc))

        # This part should ideally not be reached given the problem constraints (n, m >= 2 implies a path exists).
        # However, for completeness, if no path is found, return -1 or raise an exception.
        return -1