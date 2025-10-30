class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def distance(x1, y1, x2, y2):
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
        def intersects_or_touches(c1, c2):
            x1, y1, r1 = c1
            x2, y2, r2 = c2
            dist = distance(x1, y1, x2, y2)
            return dist <= r1 + r2
        
        def intersects_left_edge(circle):
            x, y, r = circle
            return x <= r and 0 <= y <= yCorner
        
        def intersects_right_edge(circle):
            x, y, r = circle
            return x + r >= xCorner and 0 <= y <= yCorner
        
        def intersects_bottom_edge(circle):
            x, y, r = circle
            return y <= r and 0 <= x <= xCorner
        
        def intersects_top_edge(circle):
            x, y, r = circle
            return y + r >= yCorner and 0 <= x <= xCorner
        
        n = len(circles)
        parent = list(range(n + 4))  # +4 for left, right, bottom, top edges
        
        LEFT, RIGHT, BOTTOM, TOP = n, n + 1, n + 2, n + 3
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union intersecting circles
        for i in range(n):
            for j in range(i + 1, n):
                if intersects_or_touches(circles[i], circles[j]):
                    union(i, j)
        
        # Union circles with edges
        for i in range(n):
            if intersects_left_edge(circles[i]):
                union(i, LEFT)
            if intersects_right_edge(circles[i]):
                union(i, RIGHT)
            if intersects_bottom_edge(circles[i]):
                union(i, BOTTOM)
            if intersects_top_edge(circles[i]):
                union(i, TOP)
        
        # Check if any blocking configuration exists
        if (find(LEFT) == find(RIGHT) or 
            find(BOTTOM) == find(TOP) or
            find(LEFT) == find(TOP) or
            find(LEFT) == find(BOTTOM) or
            find(RIGHT) == find(TOP) or
            find(RIGHT) == find(BOTTOM)):
            return False
        
        return True