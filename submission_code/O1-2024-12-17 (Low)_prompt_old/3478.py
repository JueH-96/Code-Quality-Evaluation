class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        We have a rectangle with corners (0,0) and (xCorner, yCorner). We want to see if there's a path
        strictly inside the rectangle (except that it may only touch the boundary at the two corners),
        avoiding all the given circles (i.e., the path cannot intersect or even touch any circle).

        Return True if such a path exists, or False otherwise.

        Key idea:
         1. If a circle covers (0,0) or (xCorner,yCorner), answer is immediately False.
         2. Otherwise, treat each circle as a "blocking region." Two circles that touch or overlap
            form a continuous blocking region. A circle also "connects" to a boundary side if it touches
            that open side (i.e., excluding the corners).
         3. If there is any chain of circles that connects:
              - left boundary to right boundary, or
              - top boundary to bottom boundary,
            then the rectangle interior is blocked in such a way that no valid path corner-to-corner can exist.
            (Because the path is forced to stay in the interior and cannot cross that chain.)
         4. If neither of these "boundary connections" occurs, then a path exists.

        We'll implement a Union-Find (Disjoint Set) over the circles plus four extra "boundary" nodes
        (TOP, BOTTOM, LEFT, RIGHT).  We unify circles that touch or overlap each other, and also unify
        any circle that touches a boundary side with that side's node.  After building these sets:
         - if TOP and BOTTOM become connected, return False
         - if LEFT and RIGHT become connected, return False
         - otherwise, return True.

        Corner coverage check:
         - If distance from (0,0) to circle center <= circle radius, corner (0,0) is blocked => False
         - If distance from (xCorner,yCorner) to circle center <= circle radius, top-right corner is blocked => False

        Touching boundary check:
         - Circle i touches left boundary (x=0) in the open segment if:
               center_x - r <= 0
             AND the circle extends in y over (0, yCorner) (so corners are excluded):
               center_y + r > 0 and center_y - r < yCorner
           Similarly for right boundary, bottom boundary, top boundary.

        Complexity:
         - We have up to n=1000 circles. Checking pairwise overlaps is O(n^2)=10^6, which is acceptable.
         - Each distance check uses 64-bit or Python int arithmetic (safe in Python).
        """

        import math

        n = len(circles)
        # Quick corner-block check
        # If (0,0) or (xCorner,yCorner) is inside/on any circle => return False immediately.
        # We'll do squared distance comparisons to avoid floats.

        def dist_sq(x1, y1, x2, y2):
            dx = x1 - x2
            dy = y1 - y2
            return dx*dx + dy*dy

        for (cx, cy, r) in circles:
            r_sq = r*r
            # Check bottom-left corner
            if dist_sq(cx, cy, 0, 0) <= r_sq:
                return False
            # Check top-right corner
            if dist_sq(cx, cy, xCorner, yCorner) <= r_sq:
                return False

        # Union-Find (Disjoint Set) for n circles + 4 boundary markers
        # We'll index the boundaries as:
        #   BOTTOM = n + 0
        #   TOP    = n + 1
        #   LEFT   = n + 2
        #   RIGHT  = n + 3

        parent = list(range(n+4))
        rank = [0]*(n+4)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                if rank[ru] < rank[rv]:
                    parent[ru] = rv
                elif rank[ru] > rank[rv]:
                    parent[rv] = ru
                else:
                    parent[rv] = ru
                    rank[ru] += 1

        BOTTOM = n+0
        TOP    = n+1
        LEFT   = n+2
        RIGHT  = n+3

        # Precompute for each circle: which boundaries does it touch (excluding corners)?
        # "Touches" a side means the circle intersects that boundary line in the open segment
        touches = [[] for _ in range(n)]  # each element is a list of boundary indices
        for i, (cx, cy, r) in enumerate(circles):
            # left boundary: x=0, open segment => y in (0, yCorner)
            # condition: center_x - r <= 0 and circle must intersect the open y-range
            if cx - r <= 0 and (cy + r > 0) and (cy - r < yCorner):
                touches[i].append(LEFT)
            # right boundary: x=xCorner, open segment => y in (0, yCorner)
            if cx + r >= xCorner and (cy + r > 0) and (cy - r < yCorner):
                touches[i].append(RIGHT)
            # bottom boundary: y=0, open segment => x in (0, xCorner)
            if cy - r <= 0 and (cx + r > 0) and (cx - r < xCorner):
                touches[i].append(BOTTOM)
            # top boundary: y=yCorner, open segment => x in (0, xCorner)
            if cy + r >= yCorner and (cx + r > 0) and (cx - r < xCorner):
                touches[i].append(TOP)

        # Union circles with the boundaries they touch
        for i in range(n):
            for b in touches[i]:
                union(i, b)

        # Now union circles that touch/overlap each other
        # Overlap or touch if dist(center1, center2) <= r1 + r2
        for i in range(n):
            cx1, cy1, r1 = circles[i]
            for j in range(i+1, n):
                cx2, cy2, r2 = circles[j]
                # Check overlap
                # dist^2 <= (r1 + r2)^2
                total_r = r1 + r2
                if dist_sq(cx1, cy1, cx2, cy2) <= total_r*total_r:
                    union(i, j)

        # After all unions, if TOP and BOTTOM are in the same set => blocked
        if find(TOP) == find(BOTTOM):
            return False
        # Or if LEFT and RIGHT are in the same set => blocked
        if find(LEFT) == find(RIGHT):
            return False

        # Otherwise, it's possible
        return True