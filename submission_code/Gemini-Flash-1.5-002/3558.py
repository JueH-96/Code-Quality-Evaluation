class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        
        q = [(0, 0, health)]
        visited = set()
        visited.add((0,0))

        while q:
            row, col, current_health = q.pop(0)

            if row == m - 1 and col == n - 1:
                return True

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < m and 0 <= new_col < n and (new_row, new_col) not in visited:
                    new_health = current_health - grid[new_row][new_col]
                    if new_health > 0:
                        q.append((new_row, new_col, new_health))
                        visited.add((new_row, new_col))
        return False