class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        def solve(row, col, current_health, visited):
            if row == m - 1 and col == n - 1:
                return current_health >= 1
            
            visited.add((row, col))
            
            # Possible moves: up, down, left, right
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n and (new_row, new_col) not in visited:
                    new_health = current_health - grid[new_row][new_col]
                    if new_health > 0:
                        if solve(new_row, new_col, new_health, visited.copy()):
                            return True
            
            return False
        
        initial_health = health - grid[0][0]
        if initial_health <= 0:
            return False
        
        return solve(0, 0, initial_health, set())