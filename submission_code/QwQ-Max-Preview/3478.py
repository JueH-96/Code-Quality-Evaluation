from typing import List
from collections import defaultdict

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start point (0,0) is inside any circle
        for x, y, r in circles:
            dx = 0 - x
            dy = 0 - y
            if dx * dx + dy * dy <= r * r:
                return False
        
        # Check if end point (xCorner, yCorner) is inside any circle
        for x, y, r in circles:
            dx = xCorner - x
            dy = yCorner - y
            if dx * dx + dy * dy <= r * r:
                return False
        
        n = len(circles)
        parent = list(range(n))
        
        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u: int, v: int) -> None:
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        
        # Precompute flags for each circle
        flags = []
        for i in range(n):
            x, y, r = circles[i]
            touches_left = (x - r) <= 0
            touches_right = (x + r) >= xCorner
            touches_bottom = (y - r) <= 0
            touches_top = (y + r) >= yCorner
            flags.append((touches_left, touches_right, touches_bottom, touches_top))
        
        # Union circles that overlap
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                sum_r = r1 + r2
                if dist_sq <= sum_r * sum_r:
                    union(i, j)
        
        # Group circles by their connected component
        components = defaultdict(list)
        for i in range(n):
            root = find(i)
            components[root].append(i)
        
        # Check each component for barriers
        for comp in components.values():
            has_left = False
            has_right = False
            has_bottom = False
            has_top = False
            for idx in comp:
                tl, tr, tb, tt = flags[idx]
                if tl:
                    has_left = True
                if tr:
                    has_right = True
                if tb:
                    has_bottom = True
                if tt:
                    has_top = True
                # Early exit if both left and right or both bottom and top are found
                if (has_left and has_right) or (has_bottom and has_top):
                    break
            if (has_left and has_right) or (has_bottom and has_top):
                return False
        
        return True