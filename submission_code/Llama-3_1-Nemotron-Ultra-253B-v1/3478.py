from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start (0,0) or end (xCorner, yCorner) is inside any circle
        for x, y, r in circles:
            # Check (0,0)
            dx = x - 0
            dy = y - 0
            if dx * dx + dy * dy <= r * r:
                return False
            # Check (xCorner, yCorner)
            dx = x - xCorner
            dy = y - yCorner
            if dx * dx + dy * dy <= r * r:
                return False
        
        n = len(circles)
        if n == 0:
            return True  # No circles, path is always possible
        
        # Initialize Union-Find with n (circles) + 4 (virtual nodes: left, right, top, bottom)
        parent = list(range(n + 4))
        rank = [1] * (n + 4)
        
        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u: int, v: int) -> None:
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            # Union by rank
            if rank[u_root] < rank[v_root]:
                parent[u_root] = v_root
            else:
                parent[v_root] = u_root
                if rank[u_root] == rank[v_root]:
                    rank[u_root] += 1
        
        # Virtual node indices
        left = n
        right = n + 1
        top = n + 2
        bottom = n + 3
        
        # Connect circles to virtual nodes based on edge contact
        for i in range(n):
            x, y, r = circles[i]
            # Check left edge (x=0)
            if x - r <= 0:
                union(i, left)
            # Check right edge (x=xCorner)
            if x + r >= xCorner:
                union(i, right)
            # Check bottom edge (y=0)
            if y - r <= 0:
                union(i, bottom)
            # Check top edge (y=yCorner)
            if y + r >= yCorner:
                union(i, top)
        
        # Check all pairs of circles for intersection and union them
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                if dist_sq <= (r1 + r2) ** 2:
                    union(i, j)
        
        # Check if left-right or top-bottom are connected through circles
        if find(left) == find(right) or find(top) == find(bottom):
            return False
        
        return True