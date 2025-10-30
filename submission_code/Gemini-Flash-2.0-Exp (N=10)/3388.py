class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        points = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    points.append((r, c))
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    r1, c1 = points[i]
                    r2, c2 = points[j]
                    r3, c3 = points[k]
                    
                    if (r1 == r2 and c1 == c3) or \
                       (r1 == r3 and c1 == c2) or \
                       (r2 == r1 and c2 == c3) or \
                       (r2 == r3 and c2 == c1) or \
                       (r3 == r1 and c3 == c2) or \
                       (r3 == r2 and c3 == c1):
                        count += 1
        return count