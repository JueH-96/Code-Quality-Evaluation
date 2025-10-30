import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        """
        Finds if a safe path exists from (0,0) to (m-1, n-1) using a
        modified Dijkstra's algorithm.
        """
        m = len(grid)
        n = len(grid[0])

        # The health must always be at least 1. This means the total loss
        # at any point cannot exceed `health - 1`.
        max_allowed_loss = health - 1

        if max_allowed_loss < 0:
            return False

        # min_loss[r][c] stores the minimum loss to reach cell (r, c)
        # on a path that satisfies the health constraint.
        min_loss = [[float('inf')] * n for _ in range(m)]
        
        # Priority queue for Dijkstra's: stores tuples of (loss, row, col).
        pq = []

        # The cost of the starting cell is incurred immediately.
        initial_loss = grid[0][0]
        if initial_loss > max_allowed_loss:
            return False
        
        min_loss[0][0] = initial_loss
        heapq.heappush(pq, (initial_loss, 0, 0))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while pq:
            current_loss, r, c = heapq.heappop(pq)

            # If we've found a better path to this cell, skip.
            if current_loss > min_loss[r][c]:
                continue

            # If the destination is reached, a valid path exists.
            if r == m - 1 and c == n - 1:
                return True

            # Explore neighbors.
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if 0 <= nr < m and 0 <= nc < n:
                    new_loss = current_loss + grid[nr][nc]

                    # A path to the neighbor is valid if:
                    # 1. The health constraint is met (new_loss <= max_allowed_loss).
                    # 2. This path offers a lower loss than any previously found path.
                    if new_loss <= max_allowed_loss and new_loss < min_loss[nr][nc]:
                        min_loss[nr][nc] = new_loss
                        heapq.heappush(pq, (new_loss, nr, nc))
                        
        # If the loop finishes, the destination was not reachable.
        return False