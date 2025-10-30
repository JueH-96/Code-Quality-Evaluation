from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])

        # The health is reduced by grid[r][c] upon entering cell (r, c).
        # We start at (0, 0), so the first health reduction happens immediately upon arrival at (0,0).
        initial_health_after_start_cell = health - grid[0][0]

        # If health is not positive after arriving at the start cell, we cannot proceed.
        # Health must be strictly positive (> 0) to make a move *from* the cell to an adjacent one.
        if initial_health_after_start_cell <= 0:
            return False

        # BFS queue stores tuples (row, col, current_health)
        # current_health is the health *after* arriving at (row, col).
        q = deque([(0, 0, initial_health_after_start_cell)])

        # max_health_reached[r][c] stores the maximum health with which cell (r, c)
        # has been visited so far. Initialize with -1, as valid health is >= 1.
        # This helps prune paths that arrive at a cell with less or equal health
        # than a previously found path.
        max_health_reached = [[-1] * n for _ in range(m)]
        max_health_reached[0][0] = initial_health_after_start_cell

        # Possible moves: up, down, left, right
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        while q:
            r, c, current_health = q.popleft()

            # Note: current_health > 0 is guaranteed because we only enqueue states
            # where health_after_move > 0. This means health is positive *before*
            # making the next move from (r, c).

            # Check if we reached the destination (m-1, n-1)
            if r == m - 1 and c == n - 1:
                # The problem requires reaching the final cell with health >= 1.
                # Since current_health is the health *after* arriving at the destination,
                # and we only enqueue states with current_health > 0, this condition is met.
                return True

            # Explore neighbors
            for i in range(4):
                nr = r + dr[i]
                nc = c + dc[i]

                # Check if the neighbor is within grid bounds
                if 0 <= nr < m and 0 <= nc < n:
                    # Calculate health upon arriving at the neighbor cell (nr, nc)
                    health_after_move = current_health - grid[nr][nc]

                    # We can only consider this path if health remains positive *after* arriving
                    # at the neighbor. If health_after_move == 0, we would be stuck there.
                    if health_after_move > 0:
                        # Check if this path to (nr, nc) is better than any previous path
                        # in terms of arriving health. We only explore if this path brings strictly
                        # more health, as it might enable reaching places previously inaccessible
                        # or reachable only with less health. If health is equal, exploring does not
                        # provide a better path in terms of health carried forward.
                        if health_after_move > max_health_reached[nr][nc]:
                            max_health_reached[nr][nc] = health_after_move
                            q.append((nr, nc, health_after_move))

        # If the queue is empty and we haven't reached the destination, no valid path exists.
        return False