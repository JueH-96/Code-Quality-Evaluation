from collections import deque

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        # First check if we lose too much health at the starting point
        initial_health = health - grid[0][0]
        if initial_health < 1:
            return False
        
        # BFS queue: (row, column, remaining health)
        queue = deque([(0, 0, initial_health)])
        
        # Track maximum health at each visited position
        visited = {(0, 0): initial_health}
        
        # Four possible directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            row, col, current_health = queue.popleft()
            
            # Check if we've reached the destination
            if row == m - 1 and col == n - 1:
                return True
            
            # Try all four directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check if new position is valid
                if 0 <= new_row < m and 0 <= new_col < n:
                    # Calculate new health after moving
                    new_health = current_health - grid[new_row][new_col]
                    
                    # Only proceed if health remains valid and better than previously found
                    if new_health >= 1 and ((new_row, new_col) not in visited or 
                                           new_health > visited[(new_row, new_col)]):
                        visited[(new_row, new_col)] = new_health
                        queue.append((new_row, new_col, new_health))
        
        # If we explored all possible paths and couldn't reach the destination
        return False