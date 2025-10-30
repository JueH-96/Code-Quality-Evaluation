import heapq
import math # For float('inf')

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])

        # dist[r][c] stores the minimum time to arrive at cell (r, c)
        # Initialize with infinity for all cells except the start
        dist = [[float('inf')] * m for _ in range(n)]

        # Start at (0, 0) at time 0
        dist[0][0] = 0

        # Priority queue stores tuples of (time, row, col), ordered by time
        # Start with the initial state
        # The tuple format for heapq is (priority, item). Here, time is the priority.
        pq = [(0, 0, 0)]

        # Directions for adjacent moves (up, down, left, right)
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]

        # Target coordinates
        target_r, target_c = n - 1, m - 1

        while pq:
            # Get the cell with the minimum arrival time found so far
            time, r, c = heapq.heappop(pq)

            # If we already found a shorter path to this cell, skip
            # This check is crucial for performance in Dijkstra when nodes can be relaxed multiple times
            if time > dist[r][c]:
                continue

            # If we reached the target, return the time
            # Since Dijkstra explores nodes in increasing order of distance (time in this case),
            # the first time we extract the target from the priority queue is the minimum time.
            if r == target_r and c == target_c:
                return time

            # Explore neighbors
            for i in range(4):
                nr = r + dr[i]
                nc = c + dc[i]

                # Check bounds
                if 0 <= nr < n and 0 <= nc < m:
                    # Calculate the earliest time we can start moving FROM (r, c) TO (nr, nc).
                    # We arrive at (r, c) at time 'time'. So we can leave (r, c) at any time >= 'time'.
                    # The constraint moveTime[nr][nc] dictates the minimum time we can *start moving TO* (nr, nc).
                    # To start the move (r, c) -> (nr, nc), both conditions must be met:
                    # 1. We must be able to leave (r, c), earliest is 'time'.
                    # 2. We must be able to start moving towards (nr, nc), earliest is moveTime[nr][nc].
                    # The earliest time we can start this move is max(time, moveTime[nr][nc]).
                    earliest_start_time_for_move = max(time, moveTime[nr][nc])

                    # The move takes 1 second. So the arrival time at (nr, nc) is 1 second after starting the move.
                    new_arrival_time = earliest_start_time_for_move + 1

                    # If this new path leads to an earlier arrival time at (nr, nc) than the current minimum
                    if new_arrival_time < dist[nr][nc]:
                        # Update the minimum arrival time for (nr, nc)
                        dist[nr][nc] = new_arrival_time
                        # Push the updated state to the priority queue
                        heapq.heappush(pq, (new_arrival_time, nr, nc))

        # This part is reached if the priority queue becomes empty and the target was not reached.
        # This would indicate the target is unreachable from the start (0,0).
        # Given the problem constraints and type, the target is expected to be reachable.
        # If unreachable, the minimum time is effectively infinity. We could return float('inf')
        # or raise an error. Returning dist[target_r][target_c] would also give float('inf') here.
        # Assuming reachability based on problem context, this line should not be hit.
        # A common practice in competitive programming might be to return -1 for unreachable,
        # but the problem asks for minimum time, so infinity or the final dist value is more accurate.
        # Let's return the final value in the distance matrix which will be infinity if unreachable.
        # Although the return inside the loop is expected for reachable targets.
        return dist[target_r][target_c] # Will be float('inf') if loop exits without hitting target