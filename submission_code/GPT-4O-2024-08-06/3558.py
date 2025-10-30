class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS queue initialized with starting position and initial health
        queue = deque([(0, 0, health)])
        visited = set((0, 0, health))
        
        while queue:
            x, y, current_health = queue.popleft()
            
            # If we reach the bottom-right corner with health >= 1, return True
            if (x, y) == (m - 1, n - 1) and current_health > 0:
                return True
            
            # Explore all possible directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check if the new position is within bounds
                if 0 <= nx < m and 0 <= ny < n:
                    # Calculate new health after moving to the new cell
                    new_health = current_health - grid[nx][ny]
                    
                    # If the new health is positive and this state hasn't been visited
                    if new_health > 0 and (nx, ny, new_health) not in visited:
                        visited.add((nx, ny, new_health))
                        queue.append((nx, ny, new_health))
        
        # If we exhaust the queue without reaching the target, return False
        return False