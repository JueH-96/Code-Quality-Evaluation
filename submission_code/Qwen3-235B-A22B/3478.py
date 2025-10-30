from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start or end points are inside any circle
        for x, y, r in circles:
            if x * x + y * y <= r * r:
                return False
            dx = xCorner - x
            dy = yCorner - y
            if dx * dx + dy * dy <= r * r:
                return False
        
        # Function to check if a circle overlaps the rectangle
        def overlaps(xc, yc, rc):
            # Closest x point on the rectangle
            if xc < 0:
                x_rect = 0
            elif xc > xCorner:
                x_rect = xCorner
            else:
                x_rect = xc
            # Closest y point on the rectangle
            if yc < 0:
                y_rect = 0
            elif yc > yCorner:
                y_rect = yCorner
            else:
                y_rect = yc
            dx = xc - x_rect
            dy = yc - y_rect
            return dx * dx + dy * dy <= rc * rc
        
        # Filter circles that overlap the rectangle
        filtered = []
        for xc, yc, rc in circles:
            if overlaps(xc, yc, rc):
                filtered.append((xc, yc, rc))
        
        if not filtered:
            return True
        
        n = len(filtered)
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pu] = pv
        
        # Build connected components
        for i in range(n):
            x1, y1, r1 = filtered[i]
            for j in range(i + 1, n):
                x2, y2, r2 = filtered[j]
                dx = x1 - x2
                dy = y1 - y2
                rs = r1 + r2
                if dx * dx + dy * dy <= rs * rs:
                    union(i, j)
        
        # Check each connected component for blocking conditions
        for i in range(n):
            if parent[i] != i:
                continue  # Skip non-root nodes
            
            left = right = bottom = top = False
            # Check all circles in this component
            for j in range(n):
                if find(j) == i:
                    x, y, r = filtered[j]
                    if x - r <= 0:
                        left = True
                    if x + r >= xCorner:
                        right = True
                    if y - r <= 0:
                        bottom = True
                    if y + r >= yCorner:
                        top = True
            
            # If component spans left-right or top-bottom, block the path
            if left and right or top and bottom:
                return False
            # Check if component blocks the diagonal from bottom-left to top-right
            # By checking combinations that could form a barrier across the diagonal
            if (left and top) or (bottom and right) or (left and bottom and top) or (top and left and right) or (bottom and left and right):
                return False
            if (bottom and top and left) or (bottom and top and right) or (left and right and top) or (left and right and bottom):
                return False
        
        return True