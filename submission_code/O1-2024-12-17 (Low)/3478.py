class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        import math
        
        # Quick check: if either corner lies inside or on any circle, return False immediately.
        # Distance from (0,0) or (xCorner,yCorner) to center must be strictly greater than radius.
        for (cx, cy, r) in circles:
            # Check (0,0)
            if (cx**2 + cy**2) <= r**2:
                return False
            # Check (xCorner,yCorner)
            dx, dy = (cx - xCorner), (cy - yCorner)
            if (dx*dx + dy*dy) <= r**2:
                return False
        
        n = len(circles)
        # We'll create a Union-Find (Disjoint Set) structure for:
        # indices [0..n-1] for the circles, plus 4 extra indices for the boundaries:
        # L = n (left), R = n+1 (right), B = n+2 (bottom), T = n+3 (top).
        L, R, B, T = n, n+1, n+2, n+3
        
        parent = list(range(n+4))
        rank = [0]*(n+4)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        
        # For each circle, check if it touches any boundary
        # A circle "touches" a boundary if center±radius crosses that boundary line.
        # If it does, union that circle with the corresponding boundary set.
        for i, (cx, cy, r) in enumerate(circles):
            if cx - r <= 0:
                union(i, L)
            if cx + r >= xCorner:
                union(i, R)
            if cy - r <= 0:
                union(i, B)
            if cy + r >= yCorner:
                union(i, T)
        
        # Now union any two circles that overlap or touch
        # i.e., distance between centers <= sum of radii
        for i in range(n):
            cx1, cy1, r1 = circles[i]
            for j in range(i+1, n):
                cx2, cy2, r2 = circles[j]
                dx = cx1 - cx2
                dy = cy1 - cy2
                # Compare squared distance to avoid float precision issues
                dist_sq = dx*dx + dy*dy
                rsum = r1 + r2
                if dist_sq <= rsum*rsum:
                    union(i, j)
        
        # If top (T) and bottom (B) end up in the same set, it means there's
        # a continuous "barrier" from top to bottom. No path from bottom-left to top-right.
        if find(T) == find(B):
            return False
        
        # Similarly, if left (L) and right (R) end up in the same set, it means
        # there's a barrier from left to right. No path from (0,0) to (xCorner,yCorner).
        if find(L) == find(R):
            return False
        
        # If neither of these barrier conditions is formed, we can travel from
        # bottom-left to top-right inside the rectangle without touching circles or edges.
        return True