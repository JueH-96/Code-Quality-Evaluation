import math
from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        We have a rectangle in the coordinate plane with bottom-left corner (0, 0)
        and top-right corner (xCorner, yCorner). We want to see if there exists a
        path strictly inside this rectangle (except for using exactly the two corners
        as start/end) that does not intersect or even touch any of the given circles.

        Equivalently, the boundary of the rectangle (except for corners) is off-limits,
        and each circle is off-limits. We must check if (0, 0) and (xCorner, yCorner)
        lie in the same connected component of the "interior" once the circles
        (as obstacles) and the rectangle boundary (except corners) are removed.

        A classical planar‐geometry fact for a convex rectangle says:
          "A path from bottom‐left (BL) to top‐right (TR) is blocked exactly if there
           is a connected 'chain' of obstacles from the other two corners (top‐left TL
           to bottom‐right BR). Any path from BL→TR must intersect any continuous
           curve from TL→BR inside a convex quadrilateral, so such an obstacle‐chain
           would forbid a crossing."

        Here, each circle can also "touch" boundary arcs.  If a circle touches in the
        open segment of the top or left edge, we regard that circle as connected to the
        "top‐left boundary" (TLB).  If it touches in the open segment of the bottom or
        right edge, we regard it as connected to the "bottom‐right boundary" (BRB).
        Two circles are mutually connected if they overlap or touch (so that there is
        no gap between them).  If we can find a chain of circles (plus those boundary
        arcs) that links TLB to BRB, that means there is effectively a fence from TL
        to BR, blocking BL→TR.

        Implementation outline:
          1) Immediately return False if (0,0) or (xCorner,yCorner) is inside or on any circle.
          2) Build a Union-Find for all circles plus two special "nodes":
             - nodeTLB = index n   (represents top + left boundary arcs)
             - nodeBRB = index n+1 (represents bottom + right boundary arcs)
          3) For each pair of circles i,j, union them if they overlap/touch:
                dist^2 <= (r_i + r_j)^2
          4) For each circle i, check if it intersects the top or left boundary arcs
             (excluding corners).  If yes, union(i, nodeTLB).
          5) Likewise, if it intersects the bottom or right boundary arcs (excluding corners),
             union(i, nodeBRB).
          6) At the end, if nodeTLB and nodeBRB end up in the same connected component,
             that means there's a chain from TL→BR, so return False. Otherwise return True.

        This method cleanly handles large coordinates (up to 1e9) and up to 1000 circles.
        """

        n = len(circles)
        # -----------------------------------------
        # 0) Quick rejection if a corner is in or on any circle
        #    "Does not touch or lie inside" => distance < = radius is forbidden
        for (cx, cy, r) in circles:
            # corner (0,0)
            dist_sq_00 = cx*cx + cy*cy
            if dist_sq_00 <= r*r:
                return False
            # corner (xCorner, yCorner)
            dx = cx - xCorner
            dy = cy - yCorner
            dist_sq_corner = dx*dx + dy*dy
            if dist_sq_corner <= r*r:
                return False

        # -----------------------------------------
        # We'll use Union-Find over n + 2 "nodes":
        # indices [0..n-1] = circles
        # index n   = top-left boundary node (TLB)
        # index n+1 = bottom-right boundary node (BRB)

        parent = list(range(n+2))
        rank   = [0]*(n+2)

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru]+=1

        # Helper to check circle overlap
        def circles_touch_or_overlap(c1, c2):
            x1, y1, r1 = c1
            x2, y2, r2 = c2
            dx = x1 - x2
            dy = y1 - y2
            rr = r1 + r2
            return (dx*dx + dy*dy) <= rr*rr

        # We want to detect intersection with open boundary segments.
        # We'll define a small helper that returns how much an interval [low, high]
        # intersects (segLow, segHigh).  If the intersection length is > 0, there's
        # an open-segment intersection (not just a corner).
        def open_segment_intersection_length(low, high, segL, segH):
            intLow = max(low, segL)
            intHigh = min(high, segH)
            return max(0.0, intHigh - intLow)

        def intersects_left_arc(cx, cy, r, yMax):
            # line x=0 in range y in (0, yMax)
            if cx > r:
                return False
            distY = math.sqrt(r*r - cx*cx) if cx < r else 0.0
            low = cy - distY
            high= cy + distY
            length = open_segment_intersection_length(low, high, 0.0, float(yMax))
            return (length > 0)

        def intersects_right_arc(cx, cy, r, xMax, yMax):
            # line x=xMax in range y in (0, yMax)
            dx = xMax - cx
            if abs(dx) > r:
                return False
            distY = math.sqrt(r*r - dx*dx)
            low = cy - distY
            high= cy + distY
            length = open_segment_intersection_length(low, high, 0.0, float(yMax))
            return (length > 0)

        def intersects_bottom_arc(cx, cy, r, xMax):
            # line y=0 in range x in (0, xMax)
            if cy > r:
                return False
            distX = math.sqrt(r*r - cy*cy) if cy < r else 0.0
            low = cx - distX
            high= cx + distX
            length = open_segment_intersection_length(low, high, 0.0, float(xMax))
            return (length > 0)

        def intersects_top_arc(cx, cy, r, yMax, xMax):
            # line y=yMax in range x in (0, xMax)
            dy = yMax - cy
            if abs(dy) > r:
                return False
            distX = math.sqrt(r*r - dy*dy)
            low = cx - distX
            high= cx + distX
            length = open_segment_intersection_length(low, high, 0.0, float(xMax))
            return (length > 0)

        # We'll define two special boundary sets:
        #   TLB = top+left boundary arcs  (index n in UF)
        #   BRB = bottom+right arcs       (index n+1 in UF)
        nodeTLB = n
        nodeBRB = n+1

        # 1) Union circles that touch or overlap
        for i in range(n):
            for j in range(i+1, n):
                if circles_touch_or_overlap(circles[i], circles[j]):
                    union(i, j)

        # 2) For each circle i, check if it intersects TLB or BRB arcs
        for i in range(n):
            cx, cy, r = circles[i]
            # TLB => top or left arcs
            if intersects_top_arc(cx, cy, r, yCorner, xCorner) or intersects_left_arc(cx, cy, r, yCorner):
                union(i, nodeTLB)
            # BRB => bottom or right arcs
            if intersects_bottom_arc(cx, cy, r, xCorner) or intersects_right_arc(cx, cy, r, xCorner, yCorner):
                union(i, nodeBRB)

        # 3) If TLB and BRB are in the same connected component, that means
        #    there's a fence from top-left corner to bottom-right corner,
        #    blocking a path from bottom-left to top-right.
        if find(nodeTLB) == find(nodeBRB):
            return False

        return True