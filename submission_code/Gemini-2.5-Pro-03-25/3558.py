import heapq
from typing import List

class Solution:
  """
  Solves the problem of finding if a safe walk exists in a grid with health constraints using Dijkstra's algorithm.
  The algorithm is adapted to maximize the remaining health at each cell rather than minimizing cost.
  """
  def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
    """
    Determines if the bottom-right corner (m-1, n-1) of the grid is reachable from the top-left corner (0, 0)
    while maintaining health strictly greater than 0 at all times.

    Args:
        grid: An m x n binary matrix where grid[i][j] = 1 indicates an unsafe cell that reduces health by 1,
              and grid[i][j] = 0 indicates a safe cell with no health cost.
        health: The initial health value when starting at (0, 0).

    Returns:
        True if a path exists to the destination (m-1, n-1) such that health remains positive (>= 1) after each step,
        False otherwise.
    """
    m = len(grid)
    n = len(grid[0])

    # Calculate the health after potentially paying the cost of the starting cell (0, 0).
    start_cost = grid[0][0]
    start_health = health - start_cost
    
    # If the health drops to 0 or below just by being at the start cell, the walk is impossible.
    if start_health <= 0:
        return False

    # max_health[r][c] stores the maximum health achieved upon arriving at cell (r, c).
    # Initialize all cells to -1, indicating they haven't been reached with positive health yet.
    # This also serves as a visited check combined with health improvement check.
    max_health = [[-1] * n for _ in range(m)]
    # Set the max health for the starting cell.
    max_health[0][0] = start_health

    # Priority queue to implement Dijkstra's algorithm.
    # Stores tuples: (-current_health, row, col).
    # We use negative health because heapq is a min-heap by default in Python,
    # and we want to prioritize states with the maximum health (effectively making it a max-heap for health).
    pq = [(-start_health, 0, 0)] 
    # heapq.heapify(pq) # Not strictly necessary for a list with a single element

    # Define the four possible directions of movement: right, left, down, up.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while pq:
        # Extract the state (cell) with the highest current health from the priority queue.
        neg_h, r, c = heapq.heappop(pq)
        current_health = -neg_h # Convert back to positive health value

        # Optimization: If the extracted state's health (current_health) is less than 
        # the maximum health already recorded for this cell (max_health[r][c]),
        # it means we've found a better path (with more health) to this cell previously.
        # Skip processing this state as it's suboptimal. This prevents cycles and redundant work.
        if current_health < max_health[r][c]:
            continue

        # Check if we have reached the destination cell (bottom-right corner).
        if r == m - 1 and c == n - 1:
            # If yes, a valid path that maintains positive health exists. Return True.
            return True

        # Explore adjacent cells (neighbors).
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Check if the neighbor cell (nr, nc) is within the grid boundaries.
            if 0 <= nr < m and 0 <= nc < n:
                # Calculate the health cost associated with moving into the neighbor cell.
                cost = grid[nr][nc]
                # Calculate the health remaining after moving into the neighbor cell.
                next_health = current_health - cost

                # Check if this move is valid and represents an improvement:
                # 1. The health must remain positive (next_health > 0) after entering the neighbor cell.
                #    (Equivalent to next_health >= 1 since health is integer).
                # 2. The health achieved at the neighbor cell via this path (next_health) must be greater than
                #    the maximum health previously recorded for this neighbor (max_health[nr][nc]).
                #    This ensures we are always finding paths that maximize health.
                if next_health > 0 and next_health > max_health[nr][nc]:
                    # If both conditions are met, update the maximum health for the neighbor cell.
                    max_health[nr][nc] = next_health
                    # Add the neighbor state to the priority queue for further exploration.
                    # Push the negative health to maintain the max-heap property based on health.
                    heapq.heappush(pq, (-next_health, nr, nc))

    # If the priority queue becomes empty and we haven't returned True,
    # it means the destination cell is unreachable while satisfying the health constraint.
    return False