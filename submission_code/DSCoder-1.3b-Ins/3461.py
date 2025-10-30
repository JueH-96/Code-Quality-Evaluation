class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        min_x, min_y = rows, cols
        max_x, max_y = 0, 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    min_x = min(min_x, i)
                    min_y = min(min_y, j)
                    max_x = max(max_x, i)
                    max_y = max(max_y, j)
        
        return (max_x - min_x + 1) * (max_y - min_y + 1)