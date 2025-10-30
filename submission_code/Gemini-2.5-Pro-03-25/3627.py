import heapq
import math
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        """
        Calculates the minimum time to reach the bottom-right cell (n-1, m-1)
        starting from the top-left cell (0, 0) at time 0 in a grid with movement time constraints.

        This problem can be modeled as finding the shortest path in a weighted graph where nodes
        are grid cells and edge weights depend on the time constraints. Dijkstra's algorithm
        is suitable for finding the shortest path in terms of time.

        Args:
            moveTime: A 2D list of size n x m, where moveTime[i][j] represents the minimum time 
                      in seconds when you can start moving TO the room (i, j).

        Returns:
            The minimum time in seconds required to reach the room (n - 1, m - 1).
        """
        n = len(moveTime)
        m = len(moveTime[0])

        # min_time[r][c] will store the minimum time to arrive at cell (r, c).
        # Initialize all arrival times to infinity, except for the starting cell.
        min_time = [[math.inf] * m for _ in range(n)]

        # Start at cell (0, 0) at time t = 0.
        min_time[0][0] = 0

        # Priority queue (min-heap) stores tuples: (arrival_time, row, col).
        # It helps process cells in the order of their earliest arrival times.
        pq = [(0, 0, 0)] 

        # Define possible movements to adjacent cells: right, left, down, up.
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while pq:
            # Extract the cell with the smallest arrival time from the priority queue.
            current_arrival_time, r, c = heapq.heappop(pq)

            # If we have already found a path to this cell with an equal or earlier arrival time,
            # the current path is not better, so we skip it. This optimization is crucial for Dijkstra's efficiency.
            if current_arrival_time > min_time[r][c]:
                continue

            # If we have reached the target cell (bottom-right corner).
            if r == n - 1 and c == m - 1:
                # Since Dijkstra's algorithm explores paths in increasing order of time,
                # the first time we reach the target, it must be via the path with the minimum time.
                return current_arrival_time 

            # Explore adjacent cells (neighbors).
            for dr, dc in directions:
                nr, nc = r + dr, c + dc # Calculate neighbor coordinates.

                # Check if the neighbor cell is within the grid boundaries.
                if 0 <= nr < n and 0 <= nc < m:
                    # Calculate the earliest time we can *start* moving from the current cell (r, c) 
                    # to the neighbor cell (nr, nc). There are two constraints:
                    # 1. We must have arrived at the current cell (r, c), so the start time must be >= current_arrival_time.
                    # 2. According to the problem, the move TO cell (nr, nc) cannot start before moveTime[nr][nc].
                    # Therefore, the actual start time is the maximum of these two lower bounds.
                    start_move_time = max(current_arrival_time, moveTime[nr][nc])
                    
                    # Moving between adjacent rooms takes exactly 1 second.
                    # The arrival time at the neighbor cell is the start time of the move plus 1 second.
                    neighbor_arrival_time = start_move_time + 1

                    # Check if the path through the current cell (r, c) offers a faster way to reach
                    # the neighbor cell (nr, nc) than any previously found path.
                    if neighbor_arrival_time < min_time[nr][nc]:
                        # Update the minimum arrival time for the neighbor cell.
                        min_time[nr][nc] = neighbor_arrival_time
                        # Add the neighbor cell to the priority queue for further exploration.
                        # It's pushed with its new earliest arrival time.
                        heapq.heappush(pq, (neighbor_arrival_time, nr, nc))
        
        # If the priority queue becomes empty and we haven't reached the target cell,
        # it implies that the target cell is unreachable under the given constraints.
        # However, based on the problem statement and typical competitive programming contexts,
        # the destination is usually guaranteed to be reachable.
        # Returning -1 could indicate an unreachable state, though this case might not occur in tests.
        return -1