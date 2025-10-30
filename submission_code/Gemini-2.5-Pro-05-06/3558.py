import heapq
from typing import List

class Solution:
  def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
    m = len(grid)
    n = len(grid[0])

    # max_health_at_cell[r][c] stores the maximum health we can have when arriving at cell (r,c).
    # Initialize with 0. Since valid health must be > 0, 0 can mean unvisited or 
    # visited with health <= 0 (which is not allowed as health must remain positive).
    max_health_at_cell = [[0 for _ in range(n)] for _ in range(m)]

    # Calculate health upon arriving at the starting cell (0,0).
    # The cost grid[0][0] (0 or 1) is paid for being at (0,0).
    start_node_health = health - grid[0][0]

    # Health must be positive (i.e., >= 1) at all times.
    # If health at (0,0) is not positive, we cannot start.
    if start_node_health <= 0:
        return False
    
    max_health_at_cell[0][0] = start_node_health
    
    # Priority queue stores tuples: (-current_health_at_node, r, c)
    # Negative health is used because heapq is a min-heap, and we want to prioritize states with maximum health.
    # This is equivalent to Dijkstra's algorithm where "distance" is negative health.
    pq = [(-start_node_health, 0, 0)]

    # Directions for movement: Right, Left, Down, Up
    # (dr, dc) format for changes in row and column indices
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while pq:
        # Pop the cell with the highest current health
        neg_h, r, c = heapq.heappop(pq)
        current_node_health = -neg_h # Convert back to positive health value

        # If a path with greater health to (r,c) has already been processed, skip this one.
        # This check is standard in Dijkstra's implementations.
        if current_node_health < max_health_at_cell[r][c]:
            continue

        # Check if the current cell (r,c) is the destination
        if r == m - 1 and c == n - 1:
            # If destination is reached, current_node_health is guaranteed to be > 0 (i.e., >= 1)
            # because of the `health_at_neighbor_node <= 0` check before pushing to PQ.
            # This satisfies the problem condition "reach the final cell with a health value of 1 or more".
            return True

        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc # Neighbor coordinates

            # Check if the neighbor is within grid boundaries
            if 0 <= nr < m and 0 <= nc < n:
                # Calculate health after moving to the neighbor cell (nr, nc).
                # The cost grid[nr][nc] (0 or 1) is paid upon entering (nr, nc).
                health_at_neighbor_node = current_node_health - grid[nr][nc]

                # Health must remain positive after moving to the neighbor.
                # If health drops to 0 or below, this path is invalid.
                if health_at_neighbor_node <= 0:
                    continue
                
                # If this path offers more health to (nr, nc) than any previously found path,
                # update max_health_at_cell and add/update neighbor in PQ.
                if health_at_neighbor_node > max_health_at_cell[nr][nc]:
                    max_health_at_cell[nr][nc] = health_at_neighbor_node
                    heapq.heappush(pq, (-health_at_neighbor_node, nr, nc))
    
    # If the priority queue becomes empty and the destination has not been reached,
    # it means the destination is unreachable while satisfying all health conditions.
    return False