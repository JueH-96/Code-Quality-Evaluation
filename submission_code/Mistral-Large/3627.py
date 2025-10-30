from collections import deque
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        queue = deque([(0, 0, 0)])  # (i, j, time)
        visited = set((0, 0))

        while queue:
            i, j, time = queue.popleft()

            # Check if we have reached the destination
            if i == n - 1 and j == m - 1:
                return time

            # Define the possible moves (right, down, left, up)
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m and (ni, nj) not in visited:
                    next_time = max(time + 1, moveTime[ni][nj])
                    queue.append((ni, nj, next_time))
                    visited.add((ni, nj))

        return -1  # In case the destination is not reachable, though the problem guarantees it is.