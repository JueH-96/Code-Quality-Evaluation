from math import sqrt
from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        # Union-Find (disjoint set) implementation.
        parent = list(range(n))
        # For every component we will record four flags:
        # touch_left, touch_right, touch_bottom, touch_top.
        touch_left   = [False]*n
        touch_right  = [False]*n
        touch_bottom = [False]*n
        touch_top    = [False]*n
        
        # Function: find with path compression.
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: 
                return
            # Merge rb into ra (arbitrary)
            parent[rb] = ra
            touch_left[ra]   = touch_left[ra]   or touch_left[rb]
            touch_right[ra]  = touch_right[ra]  or touch_right[rb]
            touch_bottom[ra] = touch_bottom[ra] or touch_bottom[rb]
            touch_top[ra]    = touch_top[ra]    or touch_top[rb]
        
        # assign each circle its boundary-touch flags.
        for i, (cx, cy, r) in enumerate(circles):
            # We say a circle touches a boundary if its disk reaches (or goes beyond) that boundary.
            # For the rectangle the “allowed” contacts are only at (0,0) and (xCorner,yCorner),
            # so any contact more than a single point counts.
            if cx - r <= 0:
                touch_left[i] = True
            if cx + r >= xCorner:
                touch_right[i] = True
            if cy - r <= 0:
                touch_bottom[i] = True
            if cy + r >= yCorner:
                touch_top[i] = True

        # For every pair of circles, if they “touch” (or overlap) in a forbidden way, union them.
        # (We use <= for circles since even tangency is not allowed.)
        for i in range(n):
            cx_i, cy_i, r_i = circles[i]
            for j in range(i+1, n):
                cx_j, cy_j, r_j = circles[j]
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                # Use squared distances to avoid overhead.
                dist2 = dx*dx + dy*dy
                # If distance between centers is <= r_i + r_j then they have a “touch”
                # (we use <= because any contact is forbidden).
                if dist2 <= (r_i + r_j)**2:
                    union(i, j)
        
        # Now, if one connected component has both a contact to one “set” of boundaries and 
        # to the complementary set, then it forms a barrier that prevents any safe path.
        # Which boundaries are on the “forbidden” side?
        # Note that (0,0) is allowed (so we do not count a contact at a single point on the left or bottom);
        # similarly, (xCorner,yCorner) is allowed. 
        #
        # A safe path from (0,0) to (xCorner,yCorner) exists if and only if there is NO connected component (of circles)
        # that simultaneously “touches” (left or bottom) AND (right or top). 
        #
        # Equivalently, if some component touches left OR bottom, we call that one “group A” (adjacent to (0,0)).
        # If some component touches right OR top, we call that “group B” (adjacent to (xCorner,yCorner)).
        # When a union–find component has both group A and group B flags true, it forms (together with the rectangle boundary)
        # a barrier that cannot be crossed by any path that stays in the open rectangle.
        
        # Check every component (by checking every circle i as a representative)
        seen = {}
        for i in range(n):
            r_i = find(i)
            if r_i not in seen:
                seen[r_i] = True
                groupA = touch_left[r_i] or touch_bottom[r_i]
                groupB = touch_right[r_i] or touch_top[r_i]
                if groupA and groupB:
                    return False
        return True