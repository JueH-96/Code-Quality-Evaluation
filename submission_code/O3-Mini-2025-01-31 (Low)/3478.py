from typing import List
import math

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
    
    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i: int, j: int):
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
            self.parent[rj] = ri

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        dsu = DSU(n)
        
        # Each circle i gets boundary flags: we'll store the flags in lists for each circle component.
        # We'll mark whether the circle touches:
        # left (L): x - r <= 0, bottom (B): y - r <= 0,
        # right (R): x + r >= xCorner, top (T): y + r >= yCorner.
        touch_left = [False] * n
        touch_bottom = [False] * n
        touch_right = [False] * n
        touch_top = [False] * n
        
        for i, (x, y, r) in enumerate(circles):
            if x - r <= 0:
                touch_left[i] = True
            if y - r <= 0:
                touch_bottom[i] = True
            if x + r >= xCorner:
                touch_right[i] = True
            if y + r >= yCorner:
                touch_top[i] = True
        
        # Build union for overlapping circles: Two circles overlap (or touch) if 
        # distance between centers <= sum of radii.
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                # Calculate squared distance to avoid sqrt
                dx = x1 - x2
                dy = y1 - y2
                if dx * dx + dy * dy <= (r1 + r2) * (r1 + r2):
                    # They touch or intersect: union them.
                    ri = dsu.find(i)
                    rj = dsu.find(j)
                    if ri != rj:
                        dsu.union(i, j)
                        newroot = dsu.find(i)
                        other = j if newroot == dsu.find(i) else i  # not used
                        # Combine boundary flags. Instead of doing in union-find structure,
                        # we do a second pass on components.
                        
        # Now, combine boundary flags per component.
        comp_left = {}
        comp_bottom = {}
        comp_right = {}
        comp_top = {}
        
        for i in range(n):
            root = dsu.find(i)
            comp_left[root] = comp_left.get(root, False) or touch_left[i]
            comp_bottom[root] = comp_bottom.get(root, False) or touch_bottom[i]
            comp_right[root] = comp_right.get(root, False) or touch_right[i]
            comp_top[root] = comp_top.get(root, False) or touch_top[i]
        
        # Now, what barriers prevent a valid path?
        # Since the path may only touch the rectangle at (0,0) and (xCorner,yCorner),
        # the remainder of the boundary is "forbidden". A valid path from (0,0) to (xCorner, yCorner)
        # cannot exist if the union of obstacles connects certain pairs of forbidden boundaries.
        # Notice that (0,0) is at the intersection of left and bottom boundaries, so aside from the corner,
        # if obstacles connect left and bottom boundaries, they block any path because one can't leave (0,0) 
        # without touching the boundary.
        # Similarly, (xCorner, yCorner) is at the intersection of right and top boundaries.
        # So if obstacles connect right and top boundaries, then the top-right corner is isolated.
        #
        # Therefore: if any component touches both left and bottom, then no valid path exists.
        # And if any component touches both right and top, then no valid path exists.
        
        for comp in comp_left:
            if comp_left[comp] and comp_bottom.get(comp, False):
                return False
            if comp_top.get(comp, False) and comp_right.get(comp, False):
                return False
        
        return True

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    print(sol.canReachCorner(3, 4, [[2,1,1]]))    # Expected True
    
    # Example 2:
    print(sol.canReachCorner(3, 3, [[1,1,2]]))    # Expected False
    
    # Example 3:
    print(sol.canReachCorner(3, 3, [[2,1,1],[1,2,1]]))  # Expected False
    
    # Example 4:
    print(sol.canReachCorner(4, 4, [[5,5,1]]))    # Expected True