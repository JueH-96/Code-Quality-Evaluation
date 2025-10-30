class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n, m = len(moveTime), len(moveTime[0])
        
        # Special case: if we can't move from start position
        if moveTime[0][1] > 1 and moveTime[1][0] > 1:
            return -1
        
        # Dijkstra's algorithm
        # (time, row, col)
        pq = [(0, 0, 0)]
        visited = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            curr_time, row, col = heapq.heappop(pq)
            
            if (row, col) in visited:
                continue
                
            visited.add((row, col))
            
            # If we reached the destination
            if row == n - 1 and col == m - 1:
                return curr_time
            
            # Try all adjacent cells
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < n and 0 <= new_col < m and 
                    (new_row, new_col) not in visited):
                    
                    # Time when we can enter the new cell
                    required_time = moveTime[new_row][new_col]
                    # We need at least curr_time + 1 (1 second to move)
                    arrival_time = curr_time + 1
                    
                    if arrival_time >= required_time:
                        # We can enter immediately
                        next_time = arrival_time
                    else:
                        # We need to wait
                        wait_time = required_time - arrival_time
                        # If wait time is even, we can waste exactly that time
                        # If wait time is odd, we need one extra second
                        if wait_time % 2 == 0:
                            next_time = required_time + 1
                        else:
                            next_time = required_time
                    
                    heapq.heappush(pq, (next_time, new_row, new_col))
        
        return -1