class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n, m = len(moveTime), len(moveTime[0])
        
        # Priority queue: (time, row, col)
        pq = [(0, 0, 0)]
        
        # Keep track of visited cells
        visited = [[False] * m for _ in range(n)]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            curr_time, row, col = heapq.heappop(pq)
            
            # If we've already visited this cell, skip
            if visited[row][col]:
                continue
            
            visited[row][col] = True
            
            # If we've reached the destination
            if row == n - 1 and col == m - 1:
                return curr_time
            
            # Try all four directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds and if not visited
                if 0 <= new_row < n and 0 <= new_col < m and not visited[new_row][new_col]:
                    # The earliest we can start moving to the new cell
                    move_start_time = max(curr_time, moveTime[new_row][new_col])
                    # Time when we arrive at the new cell
                    arrival_time = move_start_time + 1
                    
                    heapq.heappush(pq, (arrival_time, new_row, new_col))
        
        return -1  # Should never reach here given the constraints