class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS queue: each element is (row, col, current_health)
        queue = deque([(0, 0, health)])
        visited = set()
        visited.add((0, 0, health))
        
        while queue:
            r, c, current_health = queue.popleft()
            
            # If we reach the bottom-right corner with health >= 1, return True
            if r == m - 1 and c == n - 1 and current_health >= 1:
                return True
            
            # Explore all possible directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_health = current_health - grid[nr][nc]
                    if new_health > 0 and (nr, nc, new_health) not in visited:
                        visited.add((nr, nc, new_health))
                        queue.append((nr, nc, new_health))
        
        # If no valid path found
        return False