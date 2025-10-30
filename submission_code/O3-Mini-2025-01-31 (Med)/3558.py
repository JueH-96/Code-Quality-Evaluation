from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        # Calculate initial health after stepping on the starting cell.
        start_health = health - grid[0][0]
        if start_health <= 0:
            return False
        
        # Directions: up, down, left, right.
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Use BFS with state (row, column, current_health)
        dq = deque()
        dq.append((0, 0, start_health))
        
        # visited dictionary to store the best health reached at a cell.
        visited = [[-1] * n for _ in range(m)]
        visited[0][0] = start_health
        
        while dq:
            r, c, cur_health = dq.popleft()
            # Check if we've reached the bottom-right cell.
            if r == m - 1 and c == n - 1:
                # Reaching the cell is valid as long as your health is at least 1.
                if cur_health >= 1:
                    return True
            
            # Process all valid neighbors.
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    # Deduct health if the cell is unsafe.
                    next_health = cur_health - grid[nr][nc]
                    # Only step onto cell if health remains positive.
                    if next_health > 0:
                        # If we have been here with at least this much health, skip.
                        if next_health > visited[nr][nc]:
                            visited[nr][nc] = next_health
                            dq.append((nr, nc, next_health))
                            
        return False