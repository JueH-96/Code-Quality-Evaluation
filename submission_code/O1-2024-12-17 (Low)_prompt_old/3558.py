class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        
        # We need to find a path from (0,0) to (m-1, n-1) such that
        # the sum of "1"s along the path is strictly less than health.
        # That way we finish with at least 1 health point remaining.
        
        # Use BFS (or Dijkstra's-like approach) to keep track of the
        # minimum cost (sum of 1's) to reach each cell.
        # If we can reach (m-1, n-1) with total cost < health,
        # we return True, else False.
        
        # Initialize a 2D list to store the minimum damage cost at each cell.
        INF = float('inf')
        cost_grid = [[INF] * n for _ in range(m)]
        cost_grid[0][0] = grid[0][0]
        
        # Early check: if starting cell already exceeds or equals health, no path is possible.
        if cost_grid[0][0] >= health:
            return False
        
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        queue = deque()
        queue.append((0, 0))
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    new_cost = cost_grid[x][y] + grid[nx][ny]
                    # Only proceed if new_cost is better and still < health
                    if new_cost < cost_grid[nx][ny] and new_cost < health:
                        cost_grid[nx][ny] = new_cost
                        queue.append((nx, ny))
        
        # Check if we can reach the bottom-right cell with cost < health
        return cost_grid[m-1][n-1] < health