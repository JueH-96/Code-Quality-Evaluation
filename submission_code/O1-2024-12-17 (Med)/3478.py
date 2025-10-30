class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        We have a rectangle in the plane with corners (0,0) and (xCorner,yCorner). We want a path
        from (0,0) to (xCorner,yCorner) that:
          - stays strictly inside the rectangle, except at those two corners,
          - does not intersect or touch any of the given circles.

        If such a path exists, return True, otherwise return False.

        ---------------------------------------------
        IDEA / PROOF SKETCH (common "forbidden region" approach):

        1) If (0,0) or (xCorner,yCorner) is inside or on any circle, the answer is False immediately
           (no valid starting/ending point).

        2) Treat each circle as a "node" in a union-find (disjoint set). We also treat each of the four
           boundaries as "special nodes": TOP, BOTTOM, LEFT, RIGHT.

        3) We say a circle "touches" a boundary if the circle intersects or is tangent to it:
             - a circle center (cx, cy) with radius r touches the left boundary if cx - r <= 0
             - similarly for right boundary if cx + r >= xCorner
             - similarly for bottom boundary if cy - r <= 0
             - similarly for top boundary if cy + r >= yCorner

           If a circle touches a boundary, we union() that circle with the corresponding boundary node.

        4) For any two circles, if the distance between their centers <= sum of their radii, they
           overlap or are tangent, so we union() them in the DSU. In effect they form a single
           "blocking component."

        5) After processing all unions, if
             - TOP is in the same DSU set as BOTTOM, or
             - LEFT is in the same DSU set as RIGHT,
           then the rectangle's interior is "fully blocked"; there's a chain of circles (and boundaries)
           forming an impenetrable barrier from top-to-bottom or left-to-right. Hence no path
           from (0,0) to (xCorner,yCorner) can exist in the open interior, and we return False.

        6) Otherwise, return True.

        This approach works because a continuous chain of circles (plus boundary) from top to bottom
        forbids crossing the rectangle vertically, and from left to right forbids crossing it horizontally.
        A path from bottom-left to top-right must (topologically) "cross" from bottom to top and also from
        left to right. If either a vertical or horizontal chain of obstacles exists, no such path can exist.
        """

        import math

        # Quick corner check: if either corner is inside or on a circle, return False
        for (cx, cy, r) in circles:
            # Check bottom-left corner
            if (cx ** 2 + cy ** 2) <= r * r:
                return False
            # Check top-right corner
            dx = cx - xCorner
            dy = cy - yCorner
            if (dx * dx + dy * dy) <= r * r:
                return False

        n = len(circles)
        # We'll assign indices 0..n-1 for the circles
        # and n -> BOTTOM, n+1 -> TOP, n+2 -> LEFT, n+3 -> RIGHT
        BOTTOM, TOP, LEFT, RIGHT = n, n+1, n+2, n+3

        parent = list(range(n+4))  # DSU for circles + 4 boundaries

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Union circles with boundaries if they touch
        for i, (cx, cy, r) in enumerate(circles):
            if cy - r <= 0:
                union(i, BOTTOM)
            if cy + r >= yCorner:
                union(i, TOP)
            if cx - r <= 0:
                union(i, LEFT)
            if cx + r >= xCorner:
                union(i, RIGHT)

        # Union circles with each other if they overlap or touch
        for i in range(n):
            cx1, cy1, r1 = circles[i]
            for j in range(i+1, n):
                cx2, cy2, r2 = circles[j]
                dx = cx1 - cx2
                dy = cy1 - cy2
                # Use squared distance to avoid floating issues
                dist_sq = dx * dx + dy * dy
                radius_sum = r1 + r2
                if dist_sq <= radius_sum * radius_sum:
                    union(i, j)

        # If top is connected to bottom, or left is connected to right, the path is blocked
        if find(TOP) == find(BOTTOM) or find(LEFT) == find(RIGHT):
            return False
        return True