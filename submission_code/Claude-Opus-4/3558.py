class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        import heapq
        
        m, n = len(grid), len(grid[0])
        
        # Priority queue: (health_lost, row, col)
        pq = [(grid[0][0], 0, 0)]
        
        # Keep track of minimum health lost to reach each cell
        min_health_lost = {}
        min_health_lost[(0, 0)] = grid[0][0]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            health_lost, row, col = heapq.heappop(pq)
            
            # If we reached the destination
            if row == m - 1 and col == n - 1:
                # Check if we have at least 1 health remaining
                return health - health_lost >= 1
            
            # Skip if we've already found a better path to this cell
            if health_lost > min_health_lost.get((row, col), float('inf')):
                continue
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_health_lost = health_lost + grid[new_row][new_col]
                    
                    # Only proceed if this path is better than previously found paths
                    if new_health_lost < min_health_lost.get((new_row, new_col), float('inf')):
                        min_health_lost[(new_row, new_col)] = new_health_lost
                        heapq.heappush(pq, (new_health_lost, new_row, new_col))
        
        return False