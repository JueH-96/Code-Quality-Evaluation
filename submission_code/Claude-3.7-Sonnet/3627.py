from collections import deque
from typing import List

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        
        # Initialize earliest time to reach each room as infinity
        earliest_time = [[float('inf')] * m for _ in range(n)]
        earliest_time[0][0] = 0  # We start at room (0, 0) at t = 0
        
        # Queue holds (row, col, time) where time is when we reach that room
        queue = deque([(0, 0, 0)])
        
        while queue:
            row, col, t = queue.popleft()
            
            # If we've already found a better way to this room, skip
            if t > earliest_time[row][col]:
                continue
            
            # If we've reached the destination, return the time
            if row == n-1 and col == m-1:
                return t
            
            # Try moving in all four directions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new position is valid
                if 0 <= new_row < n and 0 <= new_col < m:
                    # We can only start moving to the new room at max(t, moveTime[new_row][new_col])
                    start_move_time = max(t, moveTime[new_row][new_col])
                    
                    # We reach the new room 1 second after we start moving
                    new_t = start_move_time + 1
                    
                    # If we can reach the new room earlier than before, update
                    if new_t < earliest_time[new_row][new_col]:
                        earliest_time[new_row][new_col] = new_t
                        queue.append((new_row, new_col, new_t))
        
        return earliest_time[n-1][m-1]