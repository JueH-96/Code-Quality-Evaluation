import collections
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m = len(grid)
        n = len(grid[0])

        # visited[r][c] stores the maximum health with which cell (r, c) has been reached.
        # Initialize with 0. Since health must always be >= 1, 0 effectively means "not visited
        # with positive health" or "visited with 0 or negative health".
        visited = [[0] * n for _ in range(m)]

        # Queue for BFS: stores (row, col, current_health)
        q = collections.deque()

        # Calculate initial health after "entering" the starting cell (0, 0)
        # and incurring its cost if grid[0][0] == 1.
        initial_health_at_start_cell = health - grid[0][0]

        # If health becomes non-positive at the very beginning (at (0,0) itself),
        # we cannot start with positive health.
        if initial_health_at_start_cell <= 0:
            return False

        # Add the starting cell to the queue
        q.append((0, 0, initial_health_at_start_cell))
        visited[0][0] = initial_health_at_start_cell

        # Define directions for adjacent cell moves (up, down, left, right)
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        # Perform BFS
        while q:
            r, c, current_health = q.popleft()

            # If the current cell is the destination (m-1, n-1),
            # we have found a valid path. Since health must be positive
            # to be enqueued, current_health is guaranteed to be >= 1 here.
            if r == m - 1 and c == n - 1:
                return True

            # Explore all four possible adjacent neighbors
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]

                # Check if the neighbor coordinates are within the grid boundaries
                if 0 <= nr < m and 0 <= nc < n:
                    # Calculate the health reduction for moving to the neighbor cell
                    cost_at_neighbor = grid[nr][nc]
                    new_health = current_health - cost_at_neighbor

                    # A move is valid only if:
                    # 1. Health remains strictly positive after incurring the cost (new_health > 0).
                    # 2. This path leads to the neighbor with strictly more health than
                    #    any previously found path to that cell (new_health > visited[nr][nc]).
                    #    This second condition is crucial to avoid redundant exploration and
                    #    cycles, ensuring we only explore paths that improve the health
                    #    situation at a given cell.
                    if new_health > 0 and new_health > visited[nr][nc]:
                        visited[nr][nc] = new_health
                        q.append((nr, nc, new_health))

        # If the queue becomes empty and the destination was not reached,
        # it means no valid path exists under the given constraints.
        return False