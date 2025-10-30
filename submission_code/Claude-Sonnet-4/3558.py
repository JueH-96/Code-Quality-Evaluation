class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        import heapq
        
        m, n = len(grid), len(grid[0])
        
        # Dijkstra's algorithm to find minimum health cost path
        # (health_cost, row, col)
        pq = [(grid[0][0], 0, 0)]
        
        # Track minimum health cost to reach each cell
        min_cost = {}
        min_cost[(0, 0)] = grid[0][0]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            current_cost, row, col = heapq.heappop(pq)
            
            # If we reached the destination
            if row == m - 1 and col == n - 1:
                # Check if we have enough health remaining
                return health - current_cost >= 1
            
            # Skip if we've already found a better path to this cell
            if current_cost > min_cost.get((row, col), float('inf')):
                continue
            
            # Explore all 4 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_cost = current_cost + grid[new_row][new_col]
                    
                    # Only proceed if this path is better
                    if new_cost < min_cost.get((new_row, new_col), float('inf')):
                        min_cost[(new_row, new_col)] = new_cost
                        heapq.heappush(pq, (new_cost, new_row, new_col))
        
        return False