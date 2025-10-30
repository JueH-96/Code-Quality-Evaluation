class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        import heapq
        
        n = len(moveTime)
        m = len(moveTime[0])
        
        # Priority queue: (time, row, col)
        pq = [(0, 0, 0)]
        
        # Keep track of minimum time to reach each cell
        min_time = [[float('inf')] * m for _ in range(n)]
        min_time[0][0] = 0
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while pq:
            curr_time, row, col = heapq.heappop(pq)
            
            # If we reached the destination
            if row == n - 1 and col == m - 1:
                return curr_time
            
            # Skip if we've already found a better path to this cell
            if curr_time > min_time[row][col]:
                continue
            
            # Try all four directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if the new position is valid
                if 0 <= new_row < n and 0 <= new_col < m:
                    # Time to reach the new cell is either:
                    # 1. Current time + 1 (if we can move immediately)
                    # 2. moveTime[new_row][new_col] (if we need to wait)
                    new_time = max(curr_time + 1, moveTime[new_row][new_col])
                    
                    # Only proceed if this is a better path
                    if new_time < min_time[new_row][new_col]:
                        min_time[new_row][new_col] = new_time
                        heapq.heappush(pq, (new_time, new_row, new_col))
        
        return min_time[n-1][m-1]