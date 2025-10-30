from typing import List
import heapq

class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Priority queue to store (time, x, y)
        pq = [(0, 0, 0)]  # Start at (0, 0) with time 0
        visited = set()
        
        # Directions for moving up, down, left, right
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            current_time, x, y = heapq.heappop(pq)
            
            # If we reach the bottom-right corner, return the time
            if (x, y) == (n - 1, m - 1):
                return current_time
            
            # If already visited, continue
            if (x, y) in visited:
                continue
            
            # Mark the current cell as visited
            visited.add((x, y))
            
            # Explore all adjacent cells
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in visited:
                    # Calculate the earliest time we can move to the next cell
                    next_time = max(current_time + 1, moveTime[nx][ny])
                    heapq.heappush(pq, (next_time, nx, ny))
        
        # If we exhaust the queue without reaching the target, return -1 (shouldn't happen with valid input)
        return -1