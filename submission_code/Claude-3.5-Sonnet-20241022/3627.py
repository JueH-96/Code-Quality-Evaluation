class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n, m = len(moveTime), len(moveTime[0])
        
        # Priority queue to store (time, row, col)
        pq = [(0, 0, 0)]
        # Keep track of visited cells with their minimum time
        visited = {}
        
        # Possible directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            time, row, col = heapq.heappop(pq)
            
            # If we reached the destination
            if row == n-1 and col == m-1:
                return time
            
            # If we've already found a better path to this cell
            if (row, col) in visited and visited[(row, col)] <= time:
                continue
                
            visited[(row, col)] = time
            
            # Try all possible directions
            for dx, dy in directions:
                new_row, new_col = row + dx, col + dy
                
                # Check if the new position is valid
                if 0 <= new_row < n and 0 <= new_col < m:
                    # Wait until we can enter the new room if necessary
                    wait_time = max(time, moveTime[new_row][new_col])
                    # Add 1 second for moving to the adjacent room
                    next_time = wait_time + 1
                    
                    # If we haven't visited this cell or found a better path
                    if (new_row, new_col) not in visited or visited[(new_row, new_col)] > next_time:
                        heapq.heappush(pq, (next_time, new_row, new_col))
        
        return -1  # In case destination is unreachable