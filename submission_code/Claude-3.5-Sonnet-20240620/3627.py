from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        
        # Define the possible directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Initialize the priority queue with the starting position
        pq = [(0, 0, 0)]  # (time, row, col)
        
        # Initialize visited set to keep track of visited rooms
        visited = set()
        
        while pq:
            time, row, col = heapq.heappop(pq)
            
            # If we've reached the target room, return the time
            if row == n - 1 and col == m - 1:
                return time
            
            # If we've already visited this room, skip it
            if (row, col) in visited:
                continue
            
            # Mark the current room as visited
            visited.add((row, col))
            
            # Explore adjacent rooms
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new position is within the grid
                if 0 <= new_row < n and 0 <= new_col < m:
                    # Calculate the time to move to the new room
                    new_time = max(time + 1, moveTime[new_row][new_col])
                    
                    # Add the new position to the priority queue
                    heapq.heappush(pq, (new_time, new_row, new_col))
        
        # If we can't reach the target room, return -1 (though this shouldn't happen given the constraints)
        return -1