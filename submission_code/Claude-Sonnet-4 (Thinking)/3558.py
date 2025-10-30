import heapq

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        # Dijkstra's algorithm to find minimum health cost
        pq = [(grid[0][0], 0, 0)]  # (cost, row, col)
        visited = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            cost, row, col = heapq.heappop(pq)
            
            if (row, col) in visited:
                continue
                
            visited.add((row, col))
            
            # Check if we reached the destination
            if row == m - 1 and col == n - 1:
                return health - cost >= 1
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    new_cost = cost + grid[nr][nc]
                    heapq.heappush(pq, (new_cost, nr, nc))
        
        return False