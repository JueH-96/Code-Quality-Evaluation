class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def dist_sq(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2
        
        def intersects(circle1, circle2):
            x1, y1, r1 = circle1
            x2, y2, r2 = circle2
            return dist_sq(x1, y1, x2, y2) <= (r1 + r2) ** 2
        
        def circle_touches_boundary(circle):
            x, y, r = circle
            # Check if circle touches each boundary within rectangle bounds
            left = x <= r and 0 <= y <= yCorner
            bottom = y <= r and 0 <= x <= xCorner  
            right = x + r >= xCorner and 0 <= y <= yCorner
            top = y + r >= yCorner and 0 <= x <= xCorner
            return left, bottom, right, top
        
        # Check if any circle contains the start or end corners
        for x, y, r in circles:
            if dist_sq(0, 0, x, y) <= r * r:
                return False
            if dist_sq(xCorner, yCorner, x, y) <= r * r:
                return False
        
        n = len(circles)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                if intersects(circles[i], circles[j]):
                    union(i, j)
        
        # Group circles by their connected components
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # Check if any component isolates a corner
        for group in groups.values():
            left = any(circle_touches_boundary(circles[i])[0] for i in group)
            bottom = any(circle_touches_boundary(circles[i])[1] for i in group)
            right = any(circle_touches_boundary(circles[i])[2] for i in group)
            top = any(circle_touches_boundary(circles[i])[3] for i in group)
            
            # Check if component isolates start corner (0,0)
            if left and bottom and (right or top):
                return False
            
            # Check if component isolates end corner (xCorner, yCorner)  
            if right and top and (left or bottom):
                return False
        
        return True