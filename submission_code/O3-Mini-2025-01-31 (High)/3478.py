from typing import List
import math

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    
    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa != pb:
            self.parent[pb] = pa

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Idea:
        # We have a rectangle from (0,0) to (xCorner, yCorner).
        # A valid path must lie completely in (interior) of rectangle (only endpoints may lie on boundary)
        # and must avoid (not even touch) any circle.
        # It is known that in such “obstacle‐avoidance” problems the free path exists 
        # if and only if the obstacles do NOT form a continuous chain that “blocks” the start
        # or the end. Here the start (0,0) touches the left and bottom edges (which are allowed only at (0,0))
        # and the destination (xCorner,yCorner) touches the top and right edges (allowed only at (xCorner,yCorner)).
        # In other words, if any collection of circles (touching each other) connects the left-edge (excluding (0,0))
        # with the bottom-edge (excluding (0,0)), then the start is “trapped” inside.
        # Similarly if circles connect the top-edge (excluding (xCorner,yCorner)) and the right-edge 
        # then the destination is isolated.
        #
        # We implement this idea by "union‐ing" circles that intersect (or even just touch)
        # and by union‐ing circles with virtual nodes representing the boundaries.
        # Then, if the left virtual node and bottom virtual node become connected,
        # or the top and right virtual nodes become connected, no safe path exists.
        
        n = len(circles)
        # We'll assign indices 0..n-1 for circles; then:
        LEFT_IDX   = n       # left boundary (vertical line at x=0, forbidden for all points except (0,0))
        BOTTOM_IDX = n + 1   # bottom boundary (horizontal line at y=0, forbidden except at (0,0))
        TOP_IDX    = n + 2   # top boundary (horizontal line at y=yCorner, forbidden except at (xCorner,yCorner))
        RIGHT_IDX  = n + 3   # right boundary (vertical line at x=xCorner, forbidden except at (xCorner,yCorner))
        uf = UnionFind(n + 4)
        
        # We'll define a small tolerance for floating comparisons
        TOL = 1e-9
        
        # Helper function: determine if a circle touches a given boundary segment.
        # For a vertical edge (x = edge_x), the segment is defined for y in [seg_low, seg_high].
        # For a horizontal edge (y = edge_val), the segment is defined for x in [seg_low, seg_high].
        # The parameter 'allowed' is the coordinate along the segment (y for vertical edge, x for horizontal edge)
        # at which contact is allowed (since (0,0) is allowed on left/bottom and (xCorner,yCorner) is allowed on top/right).
        def touches_edge(cx, cy, r, is_vertical, edge_coord, seg_low, seg_high, allowed):
            # For vertical edge at x = edge_coord:
            if is_vertical:
                # first, if the horizontal distance is greater than r, no contact.
                d = abs(cx - edge_coord)
                if d > r + TOL:
                    return False
                # the circle's intersection with vertical line: all y satisfying (cx - edge_coord)^2 + (y - cy)^2 <= r^2.
                # For fixed x, (y - cy)^2 <= r^2 - d^2.
                delta = math.sqrt(max(0.0, r*r - d*d))
                # The intersection interval on that vertical line is: [cy - delta, cy + delta]
                inter_low = max(seg_low, cy - delta)
                inter_high = min(seg_high, cy + delta)
                if inter_low > inter_high + TOL:
                    return False
                # If the intersection interval has positive length, we count it as touching.
                if inter_high - inter_low > TOL:
                    return True
                else:
                    # It is essentially a single point.
                    # If that point equals the allowed value then we treat it as "allowed contact" (ignore)
                    if abs(inter_low - allowed) < TOL:
                        return False
                    else:
                        return True
            else:
                # For horizontal edge at y = edge_coord:
                d = abs(cy - edge_coord)
                if d > r + TOL:
                    return False
                delta = math.sqrt(max(0.0, r*r - d*d))
                inter_low = max(seg_low, cx - delta)
                inter_high = min(seg_high, cx + delta)
                if inter_low > inter_high + TOL:
                    return False
                if inter_high - inter_low > TOL:
                    return True
                else:
                    if abs(inter_low - allowed) < TOL:
                        return False
                    else:
                        return True
        
        # For each circle, union it with the boundary virtual nodes if it "touches" the forbidden part of that edge.
        # Left edge: vertical line x=0, segment y in [0, yCorner]. Allowed contact is only at y=0.
        for i, (cx, cy, r) in enumerate(circles):
            # Left boundary:
            if touches_edge(cx, cy, r, True, 0.0, 0.0, yCorner, 0.0):
                uf.union(i, LEFT_IDX)
            # Bottom boundary: horizontal line y=0, segment x in [0, xCorner]. Allowed contact is at x=0.
            if touches_edge(cx, cy, r, False, 0.0, 0.0, xCorner, 0.0):
                uf.union(i, BOTTOM_IDX)
            # Top boundary: horizontal line y=yCorner, segment x in [0, xCorner]. Allowed contact only at x=xCorner.
            if touches_edge(cx, cy, r, False, float(yCorner), 0.0, xCorner, float(xCorner)):
                uf.union(i, TOP_IDX)
            # Right boundary: vertical line x=xCorner, segment y in [0, yCorner]. Allowed contact only at y=yCorner.
            if touches_edge(cx, cy, r, True, float(xCorner), 0.0, yCorner, float(yCorner)):
                uf.union(i, RIGHT_IDX)
        
        # Next, union circles that touch each other.
        for i in range(n):
            cx1, cy1, r1 = circles[i]
            for j in range(i+1, n):
                cx2, cy2, r2 = circles[j]
                dx = cx1 - cx2
                dy = cy1 - cy2
                # if circles touch or overlap the boundary (touching is forbidden)
                if dx*dx + dy*dy <= (r1 + r2) * (r1 + r2) + TOL:
                    uf.union(i, j)
                    
        # Now, check the two potential blocking unions.
        # The starting point (0,0) lies on the left and bottom boundaries.
        # If these boundaries are connected via circles, (0,0) is essentially “trapped”.
        if uf.find(LEFT_IDX) == uf.find(BOTTOM_IDX):
            return False
        # Similarly, the destination (xCorner, yCorner) lies on the top and right boundaries.
        if uf.find(TOP_IDX) == uf.find(RIGHT_IDX):
            return False
        return True