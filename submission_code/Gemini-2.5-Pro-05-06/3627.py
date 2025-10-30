import heapq
from typing import List # Required for List type hint

class Solution:
  def minTimeToReach(self, moveTime: List[List[int]]) -> int:
    n = len(moveTime)
    m = len(moveTime[0])

    # min_arrival_times[r][c] stores the minimum time to *arrive* at room (r,c)
    # Initialize all distances to positive infinity
    min_arrival_times = [[float('inf')] * m for _ in range(n)]

    # We start in room (0,0) at time t=0.
    # So, the arrival time at (0,0) is 0.
    min_arrival_times[0][0] = 0
    
    # Priority queue stores tuples of (time_arrived, r, c).
    # It's a min-heap, so it will always extract the state with the smallest time_arrived.
    # Python's heapq module implements a min-heap.
    # Initial state: arrived at (0,0) at time 0.
    pq = [(0, 0, 0)]  # Tuple format: (current_arrival_time_at_cell, row, col)

    # Define possible moves to adjacent cells (right, left, down, up).
    # (delta_row, delta_col)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while pq:
      current_arrival_time, r, c = heapq.heappop(pq)

      # If we've already found a shorter path to this cell (r,c),
      # this element from the priority queue is "stale", so we skip it.
      if current_arrival_time > min_arrival_times[r][c]:
        continue

      # If we have reached the destination cell (n-1, m-1):
      # Since Dijkstra's algorithm always explores paths in increasing order of cost (time),
      # the first time we extract the destination cell from the priority queue,
      # we have found the shortest path to it.
      if r == n - 1 and c == m - 1:
        return int(current_arrival_time) # Ensure int type if float('inf') was possible

      # Explore neighbors of the current cell (r,c)
      for dr, dc in directions:
        nr, nc = r + dr, c + dc # Coordinates of the neighbor cell

        # Check if the neighbor cell (nr,nc) is within the grid boundaries
        if 0 <= nr < n and 0 <= nc < m:
          # Calculate the time we would start moving from (r,c) to (nr,nc).
          # We are currently at (r,c), having arrived at current_arrival_time.
          # So, we are ready to leave (r,c) at current_arrival_time.
          # However, there's a constraint: we can only *start moving to* (nr,nc)
          # at a time T such that T >= moveTime[nr][nc].
          # Therefore, the actual time we start moving is the maximum of these two:
          time_to_start_moving = max(current_arrival_time, float(moveTime[nr][nc])) # Ensure moveTime is float for max with inf
          
          # The move itself from an adjacent cell takes 1 second.
          # So, we will arrive at the neighbor cell (nr,nc) at:
          arrival_time_at_neighbor = time_to_start_moving + 1

          # If this newly calculated arrival time at (nr,nc) is less than any
          # previously recorded arrival time for (nr,nc), it's a better path.
          if arrival_time_at_neighbor < min_arrival_times[nr][nc]:
            # Update the minimum arrival time for (nr,nc)
            min_arrival_times[nr][nc] = arrival_time_at_neighbor
            # Add this new path/state to the priority queue for further exploration
            heapq.heappush(pq, (arrival_time_at_neighbor, nr, nc))
    
    # Based on the problem's nature (grid graph, finite positive costs/times),
    # the destination should always be reachable.
    # Thus, the return statement inside the loop (when destination is popped)
    # should always be executed. This line is a fallback.
    # The problem implies the target is always reachable.
    # If it could be unreachable, this value would signify that.
    return -1 # Should not be reached in this problem as per constraints.