from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])

        # Directions for moving up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue to store (time, row, col)
        pq = [(moveTime[0][0], 0, 0)]

        # Set to store visited rooms
        visited = set()
        visited.add((0, 0))

        while pq:
            current_time, x, y = heapq.heappop(pq)

            # If we reach the bottom-right corner
            if x == n - 1 and y == m - 1:
                return current_time

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited:
                    next_time = max(current_time + 1, moveTime[nx][ny])
                    heapq.heappush(pq, (next_time, nx, ny))
                    visited.add((nx, ny))

        return -1  # This line should never be reached based on the problem constraints