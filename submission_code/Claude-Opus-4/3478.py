class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        # Union-Find with n circles + 4 edges (left, right, top, bottom)
        parent = list(range(n + 4))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Edge indices
        LEFT, RIGHT, TOP, BOTTOM = n, n + 1, n + 2, n + 3
        
        # Check if circle intersects with rectangle edge
        def intersectsEdge(x, y, r, edge):
            if edge == LEFT:
                return x - r <= 0
            elif edge == RIGHT:
                return x + r >= xCorner
            elif edge == TOP:
                return y + r >= yCorner
            elif edge == BOTTOM:
                return y - r <= 0
        
        # Check if two circles intersect or touch
        def circlesIntersect(i, j):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return dist_sq <= (r1 + r2) ** 2
        
        # Union circles with edges they intersect
        for i in range(n):
            x, y, r = circles[i]
            if intersectsEdge(x, y, r, LEFT):
                union(i, LEFT)
            if intersectsEdge(x, y, r, RIGHT):
                union(i, RIGHT)
            if intersectsEdge(x, y, r, TOP):
                union(i, TOP)
            if intersectsEdge(x, y, r, BOTTOM):
                union(i, BOTTOM)
        
        # Union intersecting circles
        for i in range(n):
            for j in range(i + 1, n):
                if circlesIntersect(i, j):
                    union(i, j)
        
        # Check if any blocking connections exist
        # Path is blocked if:
        # 1. Left and Right are connected
        # 2. Top and Bottom are connected
        # 3. Left and Top are connected (blocks upper path)
        # 4. Bottom and Right are connected (blocks lower path)
        
        if find(LEFT) == find(RIGHT):
            return False
        if find(TOP) == find(BOTTOM):
            return False
        if find(LEFT) == find(TOP):
            return False
        if find(BOTTOM) == find(RIGHT):
            return False
        
        return True