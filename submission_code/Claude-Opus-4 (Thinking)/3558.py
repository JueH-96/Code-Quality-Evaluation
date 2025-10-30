class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        import heapq
        
        m, n = len(grid), len(grid[0])
        
        # Dijkstra's algorithm to find minimum cost path
        # (cost, row, col)
        pq = [(grid[0][0], 0, 0)]
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = grid[0][0]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            cost, r, c = heapq.heappop(pq)
            
            # If we've already found a better path to this cell, skip
            if cost > dist[r][c]:
                continue
            
            # If we reached the destination
            if r == m - 1 and c == n - 1:
                # We need to have at least 1 health after visiting this cell
                return health - cost >= 1
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_cost = cost + grid[nr][nc]
                    if new_cost < dist[nr][nc]:
                        dist[nr][nc] = new_cost
                        heapq.heappush(pq, (new_cost, nr, nc))
        
        return False