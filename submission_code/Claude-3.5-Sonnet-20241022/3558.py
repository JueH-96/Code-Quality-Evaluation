class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        # Use set to keep track of visited states (row, col, health)
        visited = set()
        
        def dfs(i: int, j: int, h: int) -> bool:
            # Base cases
            if h <= 0:  # Health depleted
                return False
            if i < 0 or i >= m or j < 0 or j >= n:  # Out of bounds
                return False
            if (i, j, h) in visited:  # Already visited this state
                return False
                
            # Reduce health if current cell is unsafe
            curr_health = h - grid[i][j]
            if curr_health <= 0:
                return False
                
            # Reached destination
            if i == m-1 and j == n-1:
                return True
                
            # Mark current state as visited
            visited.add((i, j, h))
            
            # Try all four directions
            return (dfs(i+1, j, curr_health) or  # down
                    dfs(i-1, j, curr_health) or  # up
                    dfs(i, j+1, curr_health) or  # right
                    dfs(i, j-1, curr_health))    # left
        
        return dfs(0, 0, health)