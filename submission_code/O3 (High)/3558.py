from typing import List
from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        """
        0 – safe  (no health lost)
        1 – unsafe (health –1 each time the cell is entered)

        The task is equivalent to finding a path whose total number of
        unsafe cells visited (including the start and finish cells)
        does not exceed health-1.  
        We therefore search for the minimum possible “cost”
        (#unsafe cells) to reach the bottom-right corner and check
        whether it is ≤ health-1.
        """
        m, n = len(grid), len(grid[0])
        max_unsafe = health - 1                     # maximum cost we can afford

        # Cost already paid for being on the start cell
        start_cost = grid[0][0]
        if start_cost > max_unsafe:                 # out of health at the start
            return False

        INF = 10 ** 9
        dist = [[INF] * n for _ in range(m)]
        dist[0][0] = start_cost

        # 0-1 BFS: moving onto a safe cell costs 0, onto an unsafe cell costs 1
        dq = deque([(0, 0)])
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while dq:
            r, c = dq.popleft()

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_cost = dist[r][c] + grid[nr][nc]
                    if new_cost < dist[nr][nc] and new_cost <= max_unsafe:
                        dist[nr][nc] = new_cost
                        # Push to the front if the edge cost is 0, else to the back
                        if grid[nr][nc] == 0:
                            dq.appendleft((nr, nc))
                        else:
                            dq.append((nr, nc))

        # Reachable if minimum cost to destination is within our budget
        return dist[m - 1][n - 1] <= max_unsafe