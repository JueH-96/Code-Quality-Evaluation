class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        min_area = float('inf')
        
        for r1 in range(rows):
            for c1 in range(cols):
                if grid[r1][c1] == 1:
                    for r2 in range(r1, rows):
                        for c2 in range(c1, cols):
                            if grid[r2][c2] == 1:
                                if all(grid[i][j] == 1 for i in range(r1, r2+1) for j in range(c1, c2+1)):
                                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                                    min_area = min(min_area, area)
        
        return min_area